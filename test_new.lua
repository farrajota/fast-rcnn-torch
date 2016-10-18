--[[
    Test script. Computes the mAP of all proposals. 
--]]


local ffi = require 'ffi'
local tds = require 'tds'
local utils = paths.dofile('utils/init.lua')

------------------------------------------------------------------------------------------------

local function test(dataset, roi_proposals, model, modelParameters, opt)
  
  assert(dataset)
  assert(roi_proposals)
  assert(model)  
  assert(modelParameters)
  assert(opt)
  
  -- convert BatchNormalizatation backend to nn (if any)
  utils.ConvertBNcudnn2nn(model)
  
  local tester = fastrcnn.Tester(dataset, roi_proposals, model, modelParameters, opt.frcnn_test_mode or 'voc')
  
  tester:test()
end

--------------------------------------------------------------------------------

return test