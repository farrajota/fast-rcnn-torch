--[[
    Preprocess region-of-interest (RoI) data for train.
]]


local ffi = require 'ffi'
local tds = require 'tds'
local utils = paths.dofile('utils/init.lua')
local logical2indFn = utils.logical2ind

-------------------------------------------------------------------------------------------------

local function FilterRoisMinimumSize(boxes, minimum_size) 
  -- fetch widths and heights
  local boxes_width = boxes[{{},{3}}] - boxes[{{},{1}}]
  local boxes_height = boxes[{{},{4}}] - boxes[{{},{2}}]
  
  -- fetch indexes of boxes that have valid widths and heights
  local indexes = {}
  for i=1, boxes:size(1) do
    if boxes_width[i][1] >= minimum_size and boxes_height[i][1] >= minimum_size then
      table.insert(indexes, i)
    end
  end
  
  -- fetch valid indexes only
  local filtered_boxes
  if next(indexes) then
    filtered_boxes = boxes:index(1, torch.LongTensor(indexes))
  else
    filtered_boxes = torch.FloatTensor()
  end
    
  return filtered_boxes
end

-------------------------------------------------------------------------------------------------

local function ComputeBestOverlap(all_boxes, gt_boxes)
-- source: 
  local num_total_boxes = all_boxes:size(1)
  local num_gt_boxes = gt_boxes:dim() > 0 and gt_boxes:size(1) or 0
  local overlap = torch.FloatTensor(num_total_boxes,num_gt_boxes):zero()
  for idx=1, num_gt_boxes do
    local o = utils.boxoverlap(all_boxes, gt_boxes[idx])
    overlap[{{},idx}] = o
  end

  local correspondence
  if num_gt_boxes > 0 then
    overlap,correspondence = overlap:max(2)
    overlap = overlap:squeeze(2)
    correspondence = correspondence:squeeze(2)
    correspondence[overlap:eq(0)] = 0
  else
    overlap = torch.FloatTensor(num_total_boxes):zero()
    correspondence = torch.LongTensor(num_total_boxes):zero()
  end
  return overlap, correspondence
end

-------------------------------------------------------------------------------------------------

local function ComputeROIOverlapsFn(roi_boxes, data, augment_percent, quantity)
  
  -- initializations
  local objectIDList = data.filenameList.objectIDList
  local object = data.object
  local box = data.bbox
  local nFiles = data.filename:size(1)
  local nClasses = #data.classLabel
  local fileName = data.filename
  
  assert(nFiles == #roi_boxes, ('Dataset and roi proposals size mismatch: %d ~= %d'):format(nFiles, #roi_boxes))
  
  -- background class id
  local backgroundClassID = data.classID['background'] or #data.classID+1
  
  -- initialize data table for all files
  local rois_data = {}
  
  -- cycle all files and preprocess roi proposals data
  local runningIndex = 1 -- use a running index (good for datasets with files withouth ground truth boxes (data) which can be skipped/ignored during the training process)
  for ifile=1, nFiles do
    
      -- progress bar
      xlua.progress(ifile, nFiles)
      
      -- get ground truth boxes
      -- select boxes from the dataset for the current file
      local object_indexes = objectIDList[ifile][objectIDList[ifile]:gt(0)]
      if object_indexes:numel() > 0 then
          -- has boxes data
          
          local gt_boxes = {}
          local gt_classes = {}
          local filenameID = object[object_indexes[1]][1]
          local filename = ffi.string(fileName[filenameID]:data())
          local img_size = torch.LongTensor(image.load(filename):size())
          for i=1, object_indexes:size(1) do
              local box_data = box[object[object_indexes[i]][3]]
              table.insert(gt_boxes, {box_data[1], box_data[2], box_data[3], box_data[4]})
              table.insert(gt_classes, object[object_indexes[i]][2])
          end
          local num_gt_boxes = #gt_boxes
          gt_boxes = torch.FloatTensor(gt_boxes)
          gt_classes = torch.LongTensor(gt_classes)
          
          -- get proposal boxes
          local rois = FilterRoisMinimumSize(roi_boxes[ifile][{{},{1,4}}]:clone(), 5)
          
          -- merge gt boxes with proposal boxes. Ground truth boxes will be used for training and testing as proposals.
          local all_boxes
          if rois:numel()>0 then
              all_boxes = torch.cat(gt_boxes, rois, 1)
          else
              all_boxes = gt_boxes
          end
          
          -- augment the number of roi boxes (if selected)
          if augment_percent > 0 and quantity > 1 then
              all_boxes = AugmentBoxesWithJitter(all_boxes, augment_percent, quantity)
              utils.box_transform.ClampValues(all_boxes, img_size)
          end
          
          -- struct
          local rec = tds.Hash()
          
          -- set ground truth mask (1 = gt box, 0 - roi proposal box)
          local num_roi_boxes = all_boxes:size(1) - num_gt_boxes
          rec.gt_mask = torch.cat(torch.LongTensor(num_gt_boxes):fill(1), torch.LongTensor(num_roi_boxes):fill(0), 1)
          
          -- compute overlaps and correspondance
          rec.overlap_scores, rec.correspondences = ComputeBestOverlap(all_boxes, gt_boxes)
          
          rec.label = torch.LongTensor(num_roi_boxes + num_gt_boxes):fill(backgroundClassID)
          for idx=1, (num_roi_boxes + num_gt_boxes) do
              local corr = rec.correspondences[idx]
              if corr > 0 then
                  rec.label[idx] = gt_classes[corr]
              end
          end
          
          -- setup boxes storage
          rec.boxes = all_boxes
          rec.class = torch.cat(gt_classes, torch.LongTensor(num_roi_boxes):fill(backgroundClassID), 1) --background class in last position
          
          -- setup image size and image filename
          rec.image_path = filename
          rec.flipped = false 
          rec.image_size = img_size
          
          -- assign data to a table
          rois_data[runningIndex] = rec
          
          -- increment counter
          runningIndex = runningIndex + 1
          
          collectgarbage()
      else
          -- no boxes data, do nothing!
          --rois_data[ifile] = {}
      end
  end --for ifile
  
  -- output
  return rois_data
end

-------------------------------------------------------------------------------------------------

local function AddROIDataTargetsFn(rois_data, fg_thresh)
-- source: https://github.com/mahyarnajibi/fast-rcnn-torch/blob/aaa0a33805a6ca761281bde7994900127d738daa/ROI/ROI.lua#L145
  -- cycle all files
  for ifile = 1, #rois_data do
    
    -- local var
    local roidb_entry = rois_data[ifile]
    
    -- fetch ground truth boxes
    local boxes = roidb_entry.boxes
    local gt_inds = logical2indFn(roidb_entry.gt_mask)
    if gt_inds:numel() == 0 then
        return torch.FloatTensor(boxes:size(1),5):zero()
    end
    local gt_boxes = boxes:index(1,gt_inds)
    local max_overlaps = roidb_entry.overlap_scores

    -- fetch foreground boxes
    local selected_ids = logical2indFn(max_overlaps:ge(fg_thresh))
    local selected_boxes = boxes:index(1,selected_ids)

    -- Determine Targets
    local target_gts = gt_boxes:index(1,roidb_entry.correspondences:index(1,selected_ids):long())

    -- Encode the targets
    local targets = torch.FloatTensor(boxes:size(1),5):zero() -- Regression label concatenated at the end
    local encoded_selected_targets =  utils.box_transform.Transform(selected_boxes, target_gts)

    -- Concat regression targets with their labels
    local selected_labels = roidb_entry.label:index(1,selected_ids):float()
    targets:indexCopy(1,selected_ids,torch.cat(selected_labels, encoded_selected_targets,2))
    
    -- assign targets data
    rois_data[ifile].targets = targets
  end
end

-------------------------------------------------------------------------------------------------

local function GetTargetsMeanStdFn(rois_data, nClasses)
  -- Adapted from: https://github.com/mahyarnajibi/fast-rcnn-torch/blob/aaa0a33805a6ca761281bde7994900127d738daa/ROI/ROI.lua#L75-L110

  local counts = torch.zeros(nClasses, 1) + 1e-14
  local means = torch.zeros(nClasses, 4)
  local stds = torch.zeros(nClasses, 4)

  -- cycle all boxes
  for ifile=1, #rois_data do
      local cur_targets = rois_data[ifile].targets[{{},{2,5}}]
      local cur_labels = rois_data[ifile].targets[{{},{1}}]
      
      for c = 1, nClasses do
          local c_inds = logical2indFn(cur_labels:eq(c))
          if c_inds:numel()>0 then
              counts[c] = counts[c]+ c_inds:numel()
              means[c] = means[c] + cur_targets:index(1,c_inds):sum(1)
              stds[c] = stds[c] + cur_targets:index(1,c_inds):pow(2):sum(1)
          end
      end
  end

  means:cdiv(counts:expand(means:size()))
  stds= (stds:cdiv(counts:expand(stds:size())) - torch.pow(means,2)):sqrt()
  
  -- Do the normalization
  for ifile=1, #rois_data do
      local cur_targets = rois_data[ifile].targets[{{},{2,5}}]
      local cur_labels = rois_data[ifile].targets[{{},{1}}]:squeeze(2)
      for c = 1, nClasses do
          local c_inds = logical2indFn(cur_labels:eq(c))
          if c_inds:numel()>0 then
              cur_targets:indexCopy(1 , c_inds,cur_targets:index(1,c_inds) - means[c]:resize(1,means[c]:numel()):expand(torch.LongStorage{c_inds:numel(),means:size(2)}))
              cur_targets:indexCopy(1 , c_inds,cur_targets:index(1,c_inds):cdiv(stds[c]:resize(1,stds[c]:numel()):expand(torch.LongStorage{c_inds:numel(),means:size(2)})))
          end
      end
  end
  
  -- output computed means+stds
  return means, stds
end

-------------------------------------------------------------------------------------------------

local function NormalizeTargetsMeanStdFn(rois_data, nClasses, means, stds)
  -- Do the normalization
  for ifile=1, #rois_data do
      local cur_targets = rois_data[ifile].targets[{{},{2,5}}]
      local cur_labels = rois_data[ifile].targets[{{},{1}}]:squeeze(2)
      for c = 1, nClasses do
          local c_inds = logical2indFn(cur_labels:eq(c))
          if c_inds:numel()>0 then
              cur_targets:indexCopy(1 , c_inds,cur_targets:index(1,c_inds) - means[c]:resize(1,means[c]:numel()):expand(torch.LongStorage{c_inds:numel(),means:size(2)}))
              cur_targets:indexCopy(1 , c_inds,cur_targets:index(1,c_inds):cdiv(stds[c]:resize(1,stds[c]:numel()):expand(torch.LongStorage{c_inds:numel(),means:size(2)})))
          end
      end
  end
end

-------------------------------------------------------------------------------------------------

local function ComputeMeanStdBBoxreg(rois_data, fg_thresh)
  
  local regression_values = {}
  local subset_size = math.min(1000, #rois_data)

  -- cycle all boxes
  for ifile=1, subset_size do
      local cur_targets = rois_data[ifile].targets[{{},{2,5}}]
      local cur_overlap = rois_data[ifile].overlap_scores
      local inds = logical2indFn(cur_overlap:ge(fg_thresh))
      table.insert(regression_values, cur_targets:index(1,inds))
  end
  
  -- convert to tensor
  regression_values = torch.FloatTensor():cat(regression_values,1)
  
  -- output
  return {mean = regression_values:mean(1), std = regression_values:std(1)}
end

-------------------------------------------------------------------------------------------------

local function preprocessROIs(dataset, roi_data, fg_thresh, verbose)
  assert(dataset)
  assert(roi_data)
  assert(fg_thresh)
  
  local verbose = verbose or false
  
  local nClasses = #dataset.data.train.classLabel
  local data = {}
  local means, stds
  local meanstd
  for k, set in pairs({'train', 'test'}) do
      -- 1. Compute overlap, correspondences and targets
      if verbose then print('==> [1/3] Compute box overlaps and correspondences...') end
      local rois_data = ComputeROIOverlapsFn(roi_data[set], dataset.data[set], augment_percent or 1, quantity or 1)
      
      -- 2. Compute box targets
      if verbose then print('==> [2/3] Compute box targets...') end
      AddROIDataTargetsFn(rois_data, fg_thresh)
      
      --[[
      -- 3. Normalize targets wrt to the mean and std
      if set == 'train' then
          if verbose then print('==> [3/3] Compute means and stds for the targets and normalize them...') end
          means, stds = GetTargetsMeanStdFn(rois_data, nClasses)
      else
          if verbose then print('==> [3/3] Normalize targets with pre-computed means and stds...') end
          NormalizeTargetsMeanStdFn(rois_data, nClasses, means, stds)
      end
      --]]
      
      -- 3. Compute bbox mean/std
      if set == 'train' then
          if verbose then print('==> [3/3] Compute means and stds for the targets ...') end
          meanstd = ComputeMeanStdBBoxreg(rois_data, fg_thresh)
      end
      
      data[set] = {data = rois_data, means = means, stds = stds, meanstd = meanstd}
  end
  
  return data
end

-------------------------------------------------------------------------------------------------

return preprocessROIs
