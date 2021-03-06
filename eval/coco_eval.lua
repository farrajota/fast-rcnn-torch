--[[
    MSCOCO mAP evaluation function.
]]


require 'xlua'
require 'json'
local tds = require 'tds'
local coco_eval_python = require 'fastrcnn.eval.coco'


------------------------------------------------------------------------------------------------------------

local function getAboxes(res, class)
    if type(res) == 'table' or type(res) == 'cdata' then -- table or tds.hash
        if type(res[class]) == 'string' then
            return torch.load(res[class])
        else
            return res[class]
        end
    else
        error("Unknown res object: type " .. type(res))
    end
end

------------------------------------------------------------------------------------------------------------

local function convert_annotations_coco(data)
--[[ Convert tensor to table ]]
    assert(data)
    print('Converting data tensor to table format (to save as a .json file)...')
    local ann = {}
    local size = data:size(1)
    for i=1, size do
        local sample = data[i]
        local d = {
            image_id = sample[1],
            bbox = {sample[2], sample[3], sample[4], sample[5]},
            score = sample[6],
            category_id = sample[7]
        }
        table.insert(ann, d)
        if i%1000==0 or i==size then
            xlua.progress(i, size)
        end
    end
    return ann
end

------------------------------------------------------------------------------------------------------------

local function save_annotations_to_json(data, filename)
--[[ Save the annotations to a .json file ]]
    assert(data)
    assert(filename)

    local file = io.open(filename, 'w')
    file:write('[')

    local size = data:size(1)
    for i=1, size do
        local sample = data[i]
        local d = {
            image_id = sample[1],
            bbox = {sample[2], sample[3], sample[4], sample[5]},
            score = sample[6],
            category_id = sample[7]
        }

        file:write(json.encode(d))
        if i < size then
            file:write(',')
        end

        if i%1000==0 or i==size then
            xlua.progress(i, size)
        end
    end

    file:write(']')
    file:close()
end

------------------------------------------------------------------------------------------------------------

local function evaluate(loader, res, annFile)
    assert(loader)
    assert(res)
    assert(annFile)

    local nClasses = #loader.classLabel

    print("Loading files to calculate sizes...")
    local nboxes = 0
    for iclass = 1, nClasses do
        print(('Loading files for class: %d/%d'):format(iclass, nClasses))
        local aboxes = getAboxes(res, iclass)
        for _,u in pairs(aboxes) do
            if u:nDimension() > 0 then
                nboxes = nboxes + u:size(1)
            end
        end
        -- xlua.progress(iclass, nClasses)
    end
    print("Total boxes: " .. nboxes)

    local boxt = torch.FloatTensor(nboxes, 7)

    print("Loading files to create giant tensor...")
    local offset = 1
    for iclass = 1, nClasses do
        --print(('Loading files for class: %d/%d'):format(iclass, nClasses))
        local aboxes = getAboxes(res, iclass)
        for img,t in pairs(aboxes) do
            if t:nDimension() > 0 then
                local sub = boxt:narrow(1,offset,t:size(1))
                sub:select(2, 1):fill(loader.fileID(img)) -- image ID
                sub:select(2, 2):copy(t:select(2, 1) - 1) -- x1 0-indexed
                sub:select(2, 3):copy(t:select(2, 2) - 1) -- y1 0-indexed
                sub:select(2, 4):copy(t:select(2, 3) - t:select(2, 1)) -- w
                sub:select(2, 5):copy(t:select(2, 4) - t:select(2, 2)) -- h
                sub:select(2, 6):copy(t:select(2, 5)) -- score
                sub:select(2, 7):fill(loader.classID(iclass))    -- class
                offset = offset + t:size(1)
            end
        end
        xlua.progress(iclass, nClasses)
    end

    -- convert boxt to coco format
    --local ann = convert_annotations_coco(boxt)

    -- save new boxt to file
    --local tmp_annot_file = os.tmpname() .. 'coco_eval.json'
    --print('Saving result boxes to a temporary file: ' .. tmp_annot_file)
    --json.save(tmp_annot_file, ann)

    local tmp_annot_file = os.tmpname() .. 'coco_eval.json'
    print('Saving result boxes to a temporary file: ' .. tmp_annot_file)
    save_annotations_to_json(boxt, tmp_annot_file)

    boxt = nil
    collectgarbage()

    -- compute evaluation
    coco_eval_python(annFile, tmp_annot_file)

    -- remove temporary file
    os.execute('rm -rf ' .. tmp_annot_file)
end

------------------------------------------------------------------------------------------------------------

return evaluate