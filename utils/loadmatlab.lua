--[[
    Load roi proposals from matlab files.
    
    These files can contain more than one field, which allows for datasets with alot of boxes/files to be stored conveniently in a format that the package 'matio' can read (i.e., files in the format v7.0 and lower).
    
    Note: This assumes that the roi proposals are in the format [y1,x1,y2,x2] and will be converted to [x1,t1,x2,y2].
]]


local matio = require 'matio'
local dir = require 'pl.dir'
  
local function LoadMatlabFiles(path)

  assert(type(path) == 'string' or type(path) == 'table', 'Path must be either a string or a table of strings: ' .. type(path))

  if type(path) == 'table' then
    ------------------------------------------------------------------------------------------
    local function mergeTablesOfTensors(inputTable)
        -- a table of tables of tensors merges its branches (concatenates the tensors among all branches into a single branch) and returns the merged table
          local nTables = #inputTable
          if nTables > 1 then
            for ifile = 1, #inputTable[1] do
              for i=2, nTables do
                if inputTable[i][ifile]:numel() > 0 then
                  if inputTable[1][ifile]:size(2) == inputTable[i][ifile]:size(2) then
                    inputTable[1][ifile] = inputTable[1][ifile]:cat(inputTable[i][ifile],1)
                  else
                    inputTable[1][ifile] = inputTable[1][ifile][{{},{1,4}}]:cat(inputTable[i][ifile][{{},{1,4}}],1)
                    --error('Tensors dimensions size mismatch: ' .. inputTable[1][ifile]:size(2) .. '=' .. inputTable[i][ifile]:size(2))
                  end
                else
                  -- does not have any elements, skip this merge operation
                end
              end        
            end
            
            return inputTable[1]
          else
            return inputTable[1]
          end
    end -- local function
    ------------------------------------------------------------------------------------------
    
    -- get all roi boxes tensors
    local roi_boxes = {}
    for i=1, #path do
      roi_boxes[i] = LoadMatlabFiles(path[i])
    end
    
    -- merge tables together
    return mergeTablesOfTensors(roi_boxes)
  else
    local data = matio.load(path)
    
    -- check if it containers more than one field
    local counter = 0
    for k, _ in pairs(data) do
      if string.match(k, 'boxes') then
        counter = counter + 1
      end
    end
    
    local roi_boxes = {}
    if counter == 1 then
      roi_boxes = data.boxes
    else
      for i=1, counter do      
        table.addons.concat_tables(roi_boxes, data['boxes' .. i])
      end
    end
    
    -- flip to correct order ([y1,x1,y2,x2] -> [x1,y1,x2,y2])
    local range = torch.range(1,roi_boxes[1]:size(2)):long()
    range[1],range[2],range[3],range[4] = 2,1,4,3
    for ifile=1, #roi_boxes do
      if roi_boxes[ifile]:dim() > 1 then
        if roi_boxes[ifile]:numel() > 1 then
          roi_boxes[ifile] = roi_boxes[ifile]:index(2, range):float()
        else
          if roi_boxes[ifile]:sum() > 0 then 
            roi_boxes[ifile] = roi_boxes[ifile]:index(2, range):float()
          else
            roi_boxes[ifile] = torch.FloatTensor()
          end
        end
      else
        roi_boxes[ifile] = torch.FloatTensor()
      end
    end
    
    -- output boxes
    return roi_boxes
  end
end

---------------------------------------------------------------------------------------------------------------

local function LoadMatlabPath(path)
    assert(path)
    local filenames = dir.getallfiles(path)
    local roi_boxes = {}
    
    for i=1, #filenames do
        local data = matio.load(filenames[i])
        roi_boxes[i] = data
    end
    
    return roi_boxes
end

---------------------------------------------------------------------------------------------------------------

return {
    loadSingleFile = LoadMatlabFiles,
    loadMultiFiles = LoadMatlabPath
}