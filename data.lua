--[[
    Data loading/transformation functions. Loads an image and 'N' roi samples from the available dataset/rois data and outputs them alongisde with the labels, bbox shifts and the weight loss mask for optimization.
]]


function SetupDataFn(mode, rois_proprocessed, opts)
  
  local transform = paths.dofile('transform.lua')
  
  -- set local variables needed by the following functions
  -- frcnn options
  local fg_thresh = opts.frcnn_fg_thresh
  local bg_thresh_high = opts.frcnn_bg_thresh_hi
  local bg_thresh_low = opts.frcnn_bg_thresh_lo
  local roi_per_image = opts.frcnn_rois_per_img
  local num_fg_samples_per_image = opts.frcnn_num_fg_samples_per_image
  local bg_fraction = opts.frcnn_bg_fraction
  
  local backgroundID = opts.frcnn_backgroundID
  local nClasses = opts.frcnn_nClasses

  local roi_data = rois_proprocessed[mode].data
  
  -- data transform/augment function
  local transformDataFn = transform(mode, opts)

  -- utility function
  local function logical2ind(logical)
    if logical:numel() == 0 then
      return torch.LongTensor()
    end
    return torch.range(1,logical:numel())[logical:gt(0)]:long()
  end
  
  --------------------------------------------------------------------------------
  -- Select rois function
  --------------------------------------------------------------------------------

  local function SelectRoisFn(idx)
  -- fetch foreground and background samples

    -- get overlap scores
    local overlaps = roi_data[idx].overlap_scores
    -- get fg/bg indexes
    local fg_inds = logical2ind(overlaps:ge(fg_thresh)) 
    local bg_inds = logical2ind(overlaps:ge(bg_thresh_low):cmul(overlaps:lt(bg_thresh_high)))
    local bg_inds_no_overlap = logical2ind(overlaps:eq(0)) 
    
    -- determine the number of positive and negative samples for this batch
    local cur_num_fg = math.min(num_fg_samples_per_image, fg_inds:numel())
    local cur_num_bg = math.min(roi_per_image - cur_num_fg, bg_inds:numel())
    
    -- Sampling fgs without replacement
    local selected_fg_inds = torch.LongTensor()
    if cur_num_fg > 0 then
      local idx = torch.randperm(fg_inds:numel())
      selected_fg_inds = fg_inds:index(1, idx[{{1,cur_num_fg}}]:long())
    end

    -- Sampling bgs without replacement
    local selected_bg_inds = torch.LongTensor()
    if cur_num_bg > 0 then
      if bg_inds_no_overlap:numel() > 0 then
        -- get indexes for the bg_fraction with overlap
        local cur_num_bg_oe = math.ceil(cur_num_bg * bg_fraction)
        local idx1 = torch.randperm(bg_inds:numel())
        local selected_bg_oe_inds = bg_inds:index(1, idx1[{{1,cur_num_bg_oe}}]:long())
        
        -- get indexes for the bg_fraction withouth overlap with the gt boxes
        local cur_num_bg_no_oe = math.min(roi_per_image - cur_num_fg - cur_num_bg_oe, bg_inds_no_overlap:numel())
        if cur_num_bg_no_oe  > 0 then
          local idx2 = torch.randperm(bg_inds_no_overlap:numel())
          local selected_bg_no_oe_inds = bg_inds_no_overlap:index(1, idx2[{{1,cur_num_bg_no_oe}}]:long())
          selected_bg_inds = torch.cat(selected_bg_oe_inds, selected_bg_no_oe_inds,1)
        else
          selected_bg_inds = bg_inds:index(1, idx1[{{1,cur_num_bg}}]:long())
        end
      else
        local idx = torch.randperm(bg_inds:numel())
        selected_bg_inds = bg_inds:index(1, idx[{{1,cur_num_bg}}]:long())
      end
    end

    -- concatenate the sample indexes
    local batch_ids
    if selected_fg_inds:numel()>0 and selected_bg_inds:numel() > 0 then
      batch_ids = selected_fg_inds:cat(selected_bg_inds)
    elseif selected_bg_inds:numel() > 0 then
      batch_ids = selected_bg_inds
    elseif selected_fg_inds:numel() > 0 then
      batch_ids = selected_fg_inds
    else
      error('There is a sample with no positive and negative bounding boxes!')
    end
    
    -- Create the sampled batch
    local batch_rois = roi_data[idx].boxes:index(1, batch_ids)
    local batch_labels = torch.LongTensor(batch_ids:numel()):fill(backgroundID)
    batch_labels[{{1, selected_fg_inds:numel()}}] = roi_data[idx].label:index(1, selected_fg_inds)
    local batch_bbox_targets = roi_data[idx].targets:index(1, batch_ids)

    -- output
    return batch_rois, batch_labels, batch_bbox_targets
  end


  --------------------------------------------------------------------------------
  -- Fetch loss weights
  --------------------------------------------------------------------------------

  local function GetLossWeightsFn(targets)
    local loss_weights = torch.ByteTensor(targets:size(1),(nClasses+1)*4):zero()
    local labels = targets[{{},{1}}]
    local bbox_targets = targets[{{},{2,5}}]
    for i=1, bbox_targets:size(1) do
      -- select current class label id
      local cur_label = labels[i][1]
      -- check if the label does not belong to the background class
      if cur_label > 0 then
        loss_weights[{{i},{(cur_label-1)*4+1, cur_label*4}}] = 1
      end
    end
    return loss_weights, bbox_targets
  end
  
  
  --------------------------------------------------------------------------------
  -- Check batch size and padd tensor with existing fields
  --------------------------------------------------------------------------------
  
  local function CheckBatchSize(boxes, labels, targets)
    local nElems = boxes:size(1)
    
    local boxes_new = boxes:clone():resize(roi_per_image,4)
    local labels_new = labels:clone():resize(roi_per_image)
    local targets_new = targets:clone():resize(roi_per_image,5)
    
    local i=1
    for idx=nElems+1, roi_per_image do
      boxes_new[idx]:copy(boxes[i])
      labels_new[idx]=labels[i]
      targets_new[idx]:copy(targets[i])
      
      -- increment counter
      i = i+1
      
      if i > nElems then
        i=1
      end
    end
    
    return boxes_new, labels_new, targets_new
  end

  
  --------------------------------------------------------------------------------
  -- Load data function
  --------------------------------------------------------------------------------

  function loadData(idx)
    
    -- 1. Load image from file
    local img = image.load(roi_data[idx].image_path)
    
    -- 2. Get roi boxes, labels and targets
    local boxes, labels, targets = SelectRoisFn(idx)
    
    -- 3. Get loss weights
    local loss_weights, bbox_targets = GetLossWeightsFn(targets)
    
    -- 4. transform data
    local img_transf, boxes_transf = transformDataFn(img, boxes)
    
    -- 5. shuffle labes/boxes indexes
    local random_indexes = torch.randperm(boxes_transf:size(1)):long()
    boxes_transf = boxes_transf:index(1, random_indexes)
    labels = labels:index(1, random_indexes)
    bbox_targets = bbox_targets:index(1, random_indexes)
    loss_weights = loss_weights:index(1, random_indexes)
    
    -- 6. output data
    return img_transf, boxes_transf, labels, bbox_targets, loss_weights
  end

  --------------------------------------------------------------------------------

  return loadData

end