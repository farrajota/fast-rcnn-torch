--[[
    Usefull utility functions for managing networks.
]]

local tnt = require 'torchnet'

------------------------------------------------------------------------------------------------------------

local function logical2ind(logical)
  if logical:numel() == 0 then
    return torch.LongTensor()
  end
  return torch.range(1,logical:numel())[logical:gt(0)]:long()
end

------------------------------------------------------------------------------------------------------------

local function tds_to_table(input)
  assert(input)
  
  ---------------------------------------------
  local function convert_tds_to_table(input)
    assert(input)
    local out = {}
    for k, v in pairs(input) do
        out[k] = v
    end
    return out
  end
  ---------------------------------------------
  
  if type(input) == 'table' then
      return input
  elseif type(input) == 'cdata' then
      if string.lower(torch.type(input)) == 'tds.hash' or string.lower(torch.type(input)) == 'tds.hash' then
          return convert_tds_to_table(input)
      else
          error('Input must be either a tds.hash, tds.vec or a table: ' .. torch.type(input))
      end
  else
      error('Input must be either a tds.hash, tds.vec or a table: ' .. torch.type(input))
  end
end

------------------------------------------------------------------------------------------------------------

-- source: https://github.com/facebookresearch/multipathnet/blob/d677e798fcd886215b1207ae1717e2e001926b9c/utils.lua#L374
local function recursiveCopy(t1,t2)
   if torch.type(t2) == 'table' then
      t1 = (torch.type(t1) == 'table') and t1 or {t1}
      for key,_ in pairs(t2) do
         t1[key], t2[key] = recursiveCopy(t1[key], t2[key])
      end
   elseif torch.isTensor(t2) then
      t1 = torch.isTensor(t1) and t1 or t2.new()
      t1:resize(t2:size()):copy(t2)
   elseif torch.type(t2) == 'number' then
      t1 = t2
   else
      error("expecting nested tensors or tables. Got "..
      torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end

------------------------------------------------------------------------------------------------------------

-- source: https://github.com/facebookresearch/multipathnet/blob/d677e798fcd886215b1207ae1717e2e001926b9c/utils.lua#L394
local function recursiveCast(dst, src, type)
   if #dst == 0 then
      tnt.utils.table.copy(dst, nn.utils.recursiveType(src, type))
   end
   recursiveCopy(dst, src)
end

------------------------------------------------------------------------------------------------------------

local function CopySamplesDataToGPU(dst_input, dst_target, src)
  
  local nsamples = #src.input
  local max_width, max_height = 0, 0
  for i=1, nsamples do
      max_width = math.max(max_width, src.input[i][1]:size(3))
      max_height = math.max(max_height, src.input[i][1]:size(2))
  end
  
  -- copy input to GPU
  dst_input[1]:resize(nsamples,3,max_height, max_width)
  for i=1, nsamples do
      local img = src.input[i][1]
      dst_input[1][{i, {}, {1,img:size(2)}, {1,img:size(3)}}]:copy(img)
  end

  -- concatenate boxes
  local boxes, labels, bbox_targets, loss_weights
  for i=1, nsamples do
      if boxes then 
          boxes = boxes:cat(torch.FloatTensor(src.input[i][2]:size(1)):fill(i):cat(src.input[i][2],2),1)
          labels = labels:cat(src.target[i][1],1)
          bbox_targets = bbox_targets:cat(src.target[i][2][1],1)
          loss_weights = loss_weights:cat(src.target[i][2][2],1)
      else
          boxes = torch.FloatTensor(src.input[i][2]:size(1)):fill(i):cat(src.input[i][2],2)
          labels = src.target[i][1]
          bbox_targets = src.target[i][2][1]
          loss_weights = src.target[i][2][2]
      end
  end
  -- copy to GPU
  dst_input[2]:resize(boxes:size()):copy(boxes)
  dst_target[1]:resize(labels:size()):copy(labels)
  dst_target[2][1]:resize(bbox_targets:size()):copy(bbox_targets)
  dst_target[2][2]:resize(loss_weights:size()):copy(loss_weights)
  
end

------------------------------------------------------------------------------------------------------------

return {
    -- model utility functions
    model = paths.dofile('model_utils.lua'),
    
    -- non-maximum suppression
    nms = paths.dofile('nms.lua'),
    
    -- bounding box transformations
    box_transform = paths.dofile('bbox_transform.lua'),
    
    -- bounding box overlap
    boxoverlap = paths.dofile('boxoverlap.lua'),
    
    -- other functions
    logical2ind = logical2ind,
    
    -- VOC eval functions
    voc_eval = paths.dofile('voc_eval.lua'),
    
    -- convert a tds.hash/tds.vec into a table
    tds_to_table = tds_to_table,
    
    -- cast a tensor to another data type
    recursiveCast = recursiveCast,
    CopySamplesDataToGPU = CopySamplesDataToGPU,
    
    -- load matalb files
    loadmatlab = paths.dofile('loadmatlab.lua'),
    
    -- visualize detection
    visualize_detections = paths.dofile('visualize.lua'),
    
    -- store model
    --model_store = paths.dofile('store.lua'),
}