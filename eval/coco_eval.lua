--[[
    MSCOCO mAP evaluation function.
]]


local coco = paths.dofile('coco.lua')
require 'xlua'

------------------------------------------------------------------------------------------------------------

local function getAboxes()
    if type(res) == 'string' then -- res_folder
        return torch.load(('%s/%.2d.t7'):format(res, class))
    elseif type(res) == 'table' or type(res) == 'cdata' then -- table or tds.hash
        return res[class]
    else
        error("Unknown res object: type " .. type(res))
    end
end

------------------------------------------------------------------------------------------------------------

local function evaluate(dataset_name, res)
    print("Loading files to calculate sizes...")
    local nboxes = 0
    for class = 1, nClasses do
      local aboxes = getAboxes(res, class)

      for _,u in pairs(aboxes) do
        if u:nDimension() > 0 then
          nboxes = nboxes + u:size(1)
        end
      end
      -- xlua.progress(class, nClasses)
    end
    print("Total boxes: " .. nboxes)
    
    local boxt = torch.FloatTensor(nboxes, 7)

    print("Loading files to create giant tensor...")
    local offset = 1
    for class = 1, nClasses do
      local aboxes = getAboxes(res, class)
      for img,t in pairs(aboxes) do
        if t:nDimension() > 0 then
          local sub = boxt:narrow(1,offset,t:size(1))
          sub:select(2, 1):fill(image_ids[img]) -- image ID
          sub:select(2, 2):copy(t:select(2, 1) - 1) -- x1 0-indexed
          sub:select(2, 3):copy(t:select(2, 2) - 1) -- y1 0-indexed
          sub:select(2, 4):copy(t:select(2, 3) - t:select(2, 1)) -- w
          sub:select(2, 5):copy(t:select(2, 4) - t:select(2, 2)) -- h
          sub:select(2, 6):copy(t:select(2, 5)) -- score
          sub:select(2, 7):fill(dataset.data.categories.id[class])    -- class
          offset = offset + t:size(1)
        end
      end
      -- xlua.progress(class, nClasses)
    end

    local coco = Coco(annFile)
    return coco:evaluate(boxt)
end

------------------------------------------------------------------------------------------------------------

return evaluate