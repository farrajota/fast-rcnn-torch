--[[
    Test script. Computes the mAP of all proposals. 
--]]


local ffi = require 'ffi'
local tds = require 'tds'
local utils = paths.dofile('util/utils.lua')

------------------------------------------------------------------------------------------------

local function ConvertBNcudnn2nn(net)
  local function ConvertModule(net)
    return net:replace(function(x)
        if torch.type(x) == 'cudnn.BatchNormalization' then
          return cudnn.convert(x, nn)
        else
          return x
        end
    end)
  end
  net:apply(function(x) return ConvertModule(x) end)
end

------------------------------------------------------------------------------------------------

local function test(dataset, roi_proposals, model, modelParameters, opt)
  
  assert(dataset)
  assert(roi_proposals)
  assert(model)  
  assert(modelParameters)
  assert(opt)
  
  -- convert cudnn.BatchNorm modules to nn.BatchNorm (if any)
  ConvertBNcudnn2nn(model)
  
  local ImageDetector = fastrcnn.Detector(model, opt, modelParameters)
  
  -- load roi boxes from file into memory
  local roi_boxes = roi_proposals.test
  
  -- initializations
  local data = dataset.data.test
  local fileName = data.filename
  local nFiles = fileName:size(1)
  -- class names
  local classes = data.classLabel
  -- number of classes
  local nClasses = #classes
  if data.classID['background'] then
    nClasses = nClasses - 1
  end
  model:evaluate() -- set model to test mode
  
  -- (1) Select top-N boxes with the highest score per image. 
  -- NOTE: Here it is used top-100 boxes but it can be changed to use more or less boxes per image. 
  print('==> Processing dataset\'s test images + select region boxes for mAP evaluation:')
  
  -- options for mAP eval from fast-rcnn
  local nms_thresh = opt.frcnn_test_nms_thresh
  local max_per_set = 40*nFiles
  local max_per_image = 100
  local thresh = torch.ones(nClasses):mul(-math.huge)
  local scored_boxes = torch.FloatTensor() -- contains the bbox coordinates + score (to be used in the nms step)
  
  -- setup scored boxes storage
  local all_detections_boxes = tds.Hash()
  for i=1, nClasses do
    all_detections_boxes[i] = tds.Hash()
  end

  -- cycle all files and select region boxes for mAP evaluation
  for ifile=1, nFiles do
    xlua.progress(ifile, nFiles)

    -- load roi boxes
    local rois = roi_boxes[ifile]
    
    -- check if the box coordinate proposals for this file is not empty
    if rois:numel()>0 then

      -- load image
      local img = image.load(ffi.string(torch.data(fileName[ifile])))

      -- compute model output
      local scores, predicted_boxes = ImageDetector:detect(img, rois[{{},{1,4}}])
      
      -- cycle all classes
      for iclass=1, nClasses do
        local scores = scores:select(2,iclass)
       -- local scores = scores:select(2,iclass+1)
        local idx = torch.range(1,scores:numel()):long()
        local idx2 = scores:ge(thresh[iclass])
        idx = idx[idx2]
        scored_boxes:resize(idx:numel(), 5)
        if scored_boxes:numel() > 0 then
          -- use bbox predictions if exist. If not use the bbox rois coordinates.
          if predicted_boxes:numel() > 0 then
            -- use bbox predictions
            local class_boxes = predicted_boxes[{{},{(iclass-1)*4+1,(iclass)*4}}]
            scored_boxes:narrow(2,1,4):index(class_boxes,1,idx)
           -- local class_boxes = predicted_boxes[{{},{(iclass)*4+1,(iclass+1)*4}}]
           -- scored_boxes:narrow(2,1,4):index(class_boxes,1,idx)
          else
            -- use rois
            scored_boxes:narrow(2,1,4):index(rois,1,idx)
          end
          scored_boxes:select(2,5):copy(scores[idx2]) -- copy the scores
        end
        
        local keep_boxes = utils.nms(scored_boxes, nms_thresh)
        
        if keep_boxes:numel()>0 then
          local max_boxes = math.min(keep_boxes:size(1),max_per_image)
          all_detections_boxes[iclass][ifile] = keep_boxes[{{1, max_boxes},{}}]
        else
          all_detections_boxes[iclass][ifile] = torch.FloatTensor()
        end
          
        -- do some prunning
        if ifile%1000 == 0 or ifile==nFiles then
          all_detections_boxes[iclass], thresh[iclass] = utils.voc_eval.keep_top_k(all_detections_boxes[iclass], max_per_set)
        end
        
      end
    else
      for iclass=1, nClasses do
        all_detections_boxes[iclass][ifile] = torch.FloatTensor()
      end
    end --if rois:numel()>0
  
    collectgarbage()
  end --for ifile
  
  -- go back through and prune out detections below the found threshold
  for iclass = 1, nClasses do
    for ifile = 1, nFiles do
      if all_detections_boxes[iclass][ifile]:numel() > 0 then
        local I = all_detections_boxes[iclass][ifile]:select(2,5):lt(thresh[iclass])
        local idx = torch.range(1,all_detections_boxes[iclass][ifile]:size(1)):long()
        idx = idx[I]
        if idx:numel()>0 then
          all_detections_boxes[iclass][ifile] = all_detections_boxes[iclass][ifile]:index(1,idx)
        end
      end
    end
  end
  

  -- (2) Compute mAP of the selected boxes wrt the ground truth boxes from the dataset
  print('==> Computing mean average precision:')
  print('==> [class name] | [average precision]')
  local res = {}
  for iclass=1, nClasses do
    local className = classes[iclass]
    res[iclass] = utils.voc_eval.VOCevaldet(dataset.data.test, all_detections_boxes[iclass], iclass)
    print(('%s AP: %0.5f'):format(className, res[iclass]))
  end
  res = torch.Tensor(res)
  local mAP = res:mean()
  print('\n*****************')
  print(('mean AP: %0.5f'):format(mAP))
  print('*****************\n')
end

--------------------------------------------------------------------------------

return test