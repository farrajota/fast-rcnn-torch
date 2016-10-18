--[[
    Model tester class. Tests voc/coco mAP of a model on a given dataset + roi proposals.
]]


local utils = paths.dofile('utils/init.lua')

------------------------------------------------------------------------------------------------

local Tester = torch.class('fastrcnn.Tester')

function Tester:__init(dataset, roi_proposals, model, modelParameters, opt, mode)
  
  assert(model)
  assert(dataset)
  assert(rois)
  assert(mode)
  
  
  -- image detector
  model:evaluate() -- set model to test mode
  -- transformer

  
   -- initializations
  local data = dataset.data.test
  local fileName = data.filename
  local nFiles = fileName:size(1)
  -- class names
  local classes = data.classLabel
  -- number of classes
  self.nClasses = #classes
  if data.classID['background'] then
    self.nClasses = self.nClasses - 1
  end
  
  
end


function Tester:testOne(ifile)
  local dataset = self.dataset
  local thresh = self.thresh

  local img_boxes = tds.hash()
  local timer = torch.Timer()
  local timer2 = torch.Timer()
  local timer3 = torch.Timer()

  timer:reset()
  
  -- load image + boxes
  local im = image.load(ffi.string(torch.data(self.fileName[ifile])))
  local boxes = self.roi_proposals[{{ifile},{1,4}}]

  timer3:reset()
  
  -- check if proposal boxes exist
  local img_boxes, output, bbox_pred
  if boxes:numel()>0 then
      
      local all_output = {}
      local all_bbox_pred = {}

      -- detect image
      output, bbox_pred = self.ImageDetector:detect(im, boxes)

      -- clamp predictions within image
      local bbox_pred_tmp = bbox_pred:view(-1, 2)
      bbox_pred_tmp:select(2,1):clamp(1, im:size(3))
      bbox_pred_tmp:select(2,2):clamp(1, im:size(2))

      table.insert(all_output, output)
      table.insert(all_bbox_pred, bbox_pred)
      for i = 2, self.num_iter do
          -- have to copy to cuda because of torch/cutorch LongTensor differences
          self.boxselect = self.boxselect or nn.SelectBoxes():cuda()
          local new_boxes = self.boxselect:forward{output:cuda(), bbox_pred:cuda()}:float()
          output, bbox_pred = self.detec:detect(im, new_boxes, self.data_parallel_n, false)
          table.insert(all_output, output)
          table.insert(all_bbox_pred, bbox_pred)
      end

      if opt.test_use_rbox_scores then
          assert(#all_output > 1)
          -- we use the scores from iter n+1 for the boxes at iter n
          -- this means we lose one iteration worth of boxes
          table.remove(all_output, 1)
          table.remove(all_bbox_pred)
      end

      output = utils.joinTable(all_output, 1)
      bbox_pred = utils.joinTable(all_bbox_pred, 1)

      local tt2 = timer3:time().real

      timer2:reset()
      local nms_timer = torch.Timer()
      for j = 1, self.nClasses do
          local scores = output:select(2, j+1)
          local idx = torch.range(1, scores:numel()):long()
          local idx2 = scores:gt(thresh[j])
          idx = idx[idx2]
          local scored_boxes = torch.FloatTensor(idx:numel(), 5)
          if scored_boxes:numel() > 0 then
              local bx = scored_boxes:narrow(2, 1, 4)
              bx:copy(bbox_pred:narrow(2, j*4+1, 4):index(1, idx))
              scored_boxes:select(2, 5):copy(scores[idx2])
          end
          img_boxes[j] = utils.nms(scored_boxes, self.nms_thresh)
          if opt.test_bbox_voting then
              local rescaled_scored_boxes = scored_boxes:clone()
              local scores = rescaled_scored_boxes:select(2,5)
              scores:pow(opt.test_bbox_voting_score_pow or 1)

              img_boxes[j] = utils.bbox_vote(img_boxes[j], rescaled_scored_boxes, self.test_bbox_voting_nms_threshold)
          end
      end
      self.threads:synchronize()
      local nms_time = nms_timer:time().real

  else
      img_boxes = torch.FloatTensor()
      output = torch.FloatTensor()
      bbox_pred = torch.FloatTensor()
  end

  if i%1==0 and self.verbose then
      print(('test: (%s) %5d/%-5d dev: %d, forward time: %.3f, '
      .. 'select time: %.3fs, nms time: %.3fs, '
      .. 'total time: %.3fs'):format(dataset.dataset_name,
      i, dataset:size(),
      cutorch.getDevice(),
      tt2, timer2:time().real,
      nms_time, timer:time().real));
  end
  
  return img_boxes, {output, bbox_pred}
end


function Tester:test()
  self.module:evaluate()
  self.dataset:loadROIDB()

  local aboxes_t = tds.hash()

  local raw_output = tds.hash()
  local raw_bbox_pred = tds.hash()

  if not self.verbose then xlua.progress(0, self.datasetSize) end
  for ifile = 1, self.datasetSize do
      local img_boxes, raw_boxes = self:testOne(ifile)
      aboxes_t[ifile] = img_boxes
      
      --
      --if opt.test_save_raw and opt.test_save_raw ~= '' then
      --    raw_output[ifile] = raw_boxes[1]:float()
      --    raw_bbox_pred[ifile] = raw_boxes[2]:float()
      --end
      --
      if not self.verbose then xlua.progress(ifile, self.datasetSize) end
  end

  --if opt.test_save_raw and opt.test_save_raw ~= '' then
  --    torch.save(opt.test_save_raw, {raw_output, raw_bbox_pred})
  --end

  aboxes_t = self:keepTopKPerImage(aboxes_t, 100) -- coco only accepts 100/image
  local aboxes = self:transposeBoxes(aboxes_t)
  aboxes_t = nil

  return self:computeAP(aboxes)
end


function Tester:keepTopKPerImage(aboxes_t, k)
  for j = 1,self.dataset:size() do
      aboxes_t[j] = utils.keep_top_k(aboxes_t[j], k)
  end
  return aboxes_t
end


function Tester:transposeBoxes(aboxes_t)
  -- print("Running topk. max= ", self.max_per_set)
  local aboxes = tds.hash()
  for j = 1, self.num_classes do
      aboxes[j] = tds.hash()
      for i = 1, self.dataset:size() do
          aboxes[j][i] = aboxes_t[i][j]
      end
  end
  return aboxes
end

function Tester:computeAP(aboxes)
    if self.mode == 'voc' then
        --return testCoco.evaluate(self.dataset.dataset_name, aboxes)
    else
        --return testCoco.evaluate(self.dataset.dataset_name, aboxes_)
    end
end

