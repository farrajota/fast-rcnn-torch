--[[
    Load functions utilities.

    Matlab
    ======

        Load roi proposals from matlab files.

        These files can contain more than one field, which allows for datasets with alot of boxes/files to
        be stored conveniently in a format that the package 'matio' can read (i.e., files in the format v7.0 and lower).

        Note: This assumes that the roi proposals are in the format [y1,x1,y2,x2] and will be converted to [x1,t1,x2,y2].
]]


local matio = require 'matio'
local dir = require 'pl.dir'

------------------------------------------------------------------------------------------------------------

local function LoadMatlabFiles(path)
    assert(path)
    local type_input = type(path)
    assert(type_input == 'string' or type_input == 'table', 'Path must be either a string or a table of strings: ' .. type(path))

    if type_input == 'table' then
        -- get all roi boxes tensors
        local roi_boxes = {}
        for i=1, #path do
            table.insert(roi_boxes, LoadMatlabFiles(path[i]))
        end
        -- merge tables together
        return torch.FloatTensor():cat(roi_boxes,1)
    else
        local data = matio.load(path)
        assert(data)

        -- check if it containers more than one field
        local counter = 0
        for k, _ in pairs(data) do
            if string.match(k, 'boxes') then
                counter = counter + 1
            end
        end
        assert(counter > 0, 'File doesn\'t contain any bbox proposals data: ' .. path)

        local roi_boxes = {}
        if counter == 1 then
            roi_boxes = data.boxes
        else
            for i=1, counter do
                table.insert(roi_boxes, data['boxes' .. i])
            end
            roi_boxes = torch.FloatTensor():cat(roi_boxes,1)
        end

        if type(roi_boxes) == 'table' then
            -- flip to correct order ([y1,x1,y2,x2] -> [x1,y1,x2,y2])
            local range = torch.range(1,roi_boxes[1]:size(2)):long() -- might have more :size(2) > 4
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
        else
            local range = torch.range(1,roi_boxes:size(2)):long() -- might have more :size(2) > 4
            range[1],range[2],range[3],range[4] = 2,1,4,3
            roi_boxes = roi_boxes:index(2, range):float()
        end

        -- output boxes
        return roi_boxes
    end
end

------------------------------------------------------------------------------------------------------------

local function LoadMatlabPath(path)
    assert(path)
    local filenames = dir.getallfiles(path)
    local roi_boxes = {}
    for i=1, #filenames do
        local data = matio.load(filenames[i])
        assert(data, 'File doesn\'t contain any bbox proposals data: ' .. filenames[i])
        roi_boxes[i] = data
    end
    return roi_boxes
end

------------------------------------------------------------------------------------------------------------

return {
    matlab = {
        single = LoadMatlabFiles,
        multi = LoadMatlabPath
    }
}