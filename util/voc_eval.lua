--[[
    VOC evaluation functions (mean Average Precision [mAP]).
]]


local boxoverlap = paths.dofile('boxoverlap.lua')

---------------------------------------------------------------------------------------------------------------------

local function joinTable(input,dim)
  local size = torch.LongStorage()
  local is_ok = false
  for i=1,#input do
    local currentOutput = input[i]
    if currentOutput:numel() > 0 then
      if not is_ok then
        size:resize(currentOutput:dim()):copy(currentOutput:size())
        is_ok = true
      else
        size[dim] = size[dim] + currentOutput:size(dim)
      end    
    end
  end
  local output = input[1].new():resize(size)
  local offset = 1
  for i=1,#input do
    local currentOutput = input[i]
    if currentOutput:numel() > 0 then
      output:narrow(dim, offset,
                    currentOutput:size(dim)):copy(currentOutput)
      offset = offset + currentOutput:size(dim)
    end
  end
  return output
end

---------------------------------------------------------------------------------------------------------------------

local function keep_top_k(boxes,top_k)
-- source: https://github.com/fmassa/object-detection.torch/blob/master/utils.lua#L34
  local X = joinTable(boxes,1)
  if X:numel() == 0 then
    return
  end
  local scores = X[{{},-1}]:sort(1,true)
  local thresh = scores[math.min(scores:numel(),top_k)]
  for i=1,#boxes do
    local bbox = boxes[i]
    if bbox:numel() > 0 then
      local idx = torch.range(1,bbox:size(1)):long()
      local keep = bbox[{{},-1}]:ge(thresh)
      idx = idx[keep]
      if idx:numel() > 0 then
        boxes[i] = bbox:index(1,idx)
      else
        boxes[i]:resize()
      end
    end
  end
  return boxes, thresh
end

---------------------------------------------------------------------------------------------------------------------

local function VOCap(rec,prec)
-- source: https://github.com/fmassa/object-detection.torch/blob/master/utils.lua#L61
-- compute average precision
  local ap = 0
  for t=0,1,0.1 do
    local c = prec[rec:ge(t)]
    local p
    if c:numel() > 0 then
      p = torch.max(c)
    else
      p = 0
    end
    ap=ap+p/11
  end
  return ap
end

---------------------------------------------------------------------------------------------------------------------

local function VOCevaldet(data, scored_boxes, classID)
-- adapted from: https://github.com/fmassa/object-detection.torch/blob/master/utils.lua#L128
-- compute average precision (AP), recall and precision
  
  local objectIDList = data.filenameList.objectIDList
  local object = data.object            -- object pointer
  local box = data.bbox                 -- bounding box pointer
  local nFiles = data.filename:size(1)  -- number of files
    
  local num_pr = 0
  local energy = {}
  local correct = {}
  
  local count = 0
  
  for ifile=1, nFiles do
    -- fetch all bboxes belonging to this file and for this classID
    local bbox = {}
    local det = {}
    
    local object_indexes = objectIDList[ifile][objectIDList[ifile]:gt(0)]
    if object_indexes:numel() > 0 then
      for ibb=1, object_indexes:size(1) do
        --local bbox_data = bboxList[bbox_indexes[ibb]]
        local bbox_data = box[object[object_indexes[ibb]][3]]
        if object[object_indexes[ibb]][2] == classID then
          table.insert(bbox,{bbox_data[1], bbox_data[2], bbox_data[3], bbox_data[4]})
          table.insert(det, 0)
          count = count + 1
        end
      end
    end
    
    bbox = torch.Tensor(bbox)
    det = torch.Tensor(det)
    
    local num = scored_boxes[ifile]:numel()>0 and scored_boxes[ifile]:size(1) or 0
    for j=1, num do
      local bbox_pred = scored_boxes[ifile][j]
      num_pr = num_pr + 1
      table.insert(energy,bbox_pred[5])
      
      if bbox:numel()>0 then
        local o = boxoverlap(bbox,bbox_pred[{{1,4}}])
        local maxo,index = o:max(1)
        maxo = maxo[1]
        index = index[1]
        if maxo >=0.5 and det[index] == 0 then
          correct[num_pr] = 1
          det[index] = 1
        else
          correct[num_pr] = 0
        end
      else
          correct[num_pr] = 0        
      end
    end
    
  end
  
  if #energy == 0 then
    return 0,torch.Tensor(),torch.Tensor()
  end
  
  energy = torch.Tensor(energy)
  correct = torch.Tensor(correct)
  
  local threshold,index = energy:sort(true)

  correct = correct:index(1,index)

  -- compute recall + precision
  local n = threshold:numel()
  
  local recall = torch.zeros(n)
  local precision = torch.zeros(n)

  local num_correct = 0

  for i = 1,n do
      --compute precision
      local num_positive = i
      num_correct = num_correct + correct[i]
      if num_positive ~= 0 then
          precision[i] = num_correct / num_positive;
      else
          precision[i] = 0;
      end
      
      --compute recall
      recall[i] = num_correct / count
  end

  -- compute average precision
  local ap = VOCap(recall, precision)

  -- outputs
  return ap, recall, precision
end

---------------------------------------------------------------------------------------------------------------------

return {
    boxoverlap = boxoverlap,
    keep_top_k = keep_top_k,
    VOCevaldet = VOCevaldet
}