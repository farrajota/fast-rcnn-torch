--[[
    Utility functions.
]]


local function logical2ind(logical)
    if logical:numel() == 0 then
        return torch.LongTensor()
    end
    return torch.range(1,logical:numel())[logical:gt(0)]:long()
end

------------------------------------------------------------------------------------------------------------

local function CopySamplesDataToGPU(dst_input, dst_target, src)
    assert(dst_input)
    assert(dst_target)
    assert(src)

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
    model = require 'utils.model',

    -- non-maximum suppression
    nms = require 'utils.nms',

    -- bounding box overlap
    box = require 'utils.box',

    -- table functions
    table = require 'utils.table',

    -- load matlab files
    load = require 'utils.load',

    -- visualize object detections with a window
    visualize_detections = require 'utils.visualize',

    -- other functions
    logical2ind = logical2ind,

    -- cast a tensor to another data type
    CopySamplesDataToGPU = CopySamplesDataToGPU,
}