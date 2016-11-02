--[[
    Fast-RCNN class. Has train, test and detect functions.
]]

require 'torch'
require 'xlua'
require 'optim'
require 'nn'
require 'nngraph'
require 'string'
require 'image'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'inn'

fastrcnn = {}

-- modules
paths.dofile('modules/BBoxNorm.lua')
paths.dofile('modules/NoBackprop.lua')
paths.dofile('modules/BBoxRegressionCriterion.lua')

-- frcnn classes
paths.dofile('BatchROISampler.lua') -- data loader/generator
paths.dofile('ImageDetector.lua')   -- single image detector/tester
paths.dofile('Tester.lua')          -- dataset tester
paths.dofile('Options.lua')         -- Fast-RCNN options parser

-- load setup/options functions
fastrcnn.train = paths.dofile('train.lua')
fastrcnn.test = paths.dofile('test.lua')
fastrcnn.utils = paths.dofile('utils/init.lua')
fastrcnn.visualize_detections = paths.dofile('utils/visualize.lua')

return fastrcnn