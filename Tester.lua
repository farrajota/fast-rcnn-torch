--[[
    Model tester class. Tests voc/coco mAP of a model on a given dataset + roi proposals.
]]


local ffi = require 'ffi'

------------------------------------------------------------------------------------------------

local tester = torch.class('fastrcnn.Tester')

function tester:__init(model, dataset, rois, mode)
  
  assert(model)
  assert(dataset)
  assert(rois)
  assert(mode)
  
  
  
end

function tester:testOne()
end

function tester:test()
end

