--[[
    Evaluation functions.
]]

return{
    coco = paths.dofile('coco_eval.lua'),
    pascal = paths.dofile('pascal_eval.lua'),
    --coco = require 'fastrcnn.eval.coco_eval',
    --pascal = require 'fastrcnn.eval.pascal_eval'
}