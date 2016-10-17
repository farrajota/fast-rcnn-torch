--[[
    Usefull utility functions for managing networks.
]]


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
    
    
    -- load matalb files
    loadmatlab = paths.dofile('loadmatlab.lua'),
    
    -- visualize detection
    visualize_detections = paths.dofile('visualize.lua'),
    
    -- store model
    --model_store = paths.dofile('store.lua'),
}