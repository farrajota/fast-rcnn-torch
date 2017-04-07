--[[
    Fast-RCNN class. Has train, test and detect functions.
]]

require 'paths'
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
--[[
require 'fastrcnn.modules'         -- bbox modules for train/test
require 'fastrcnn.BatchROISampler' -- data loader/generator
require 'fastrcnn.ImageDetector'   -- single image detector/tester
require 'fastrcnn.Tester'          -- dataset tester
require 'fastrcnn.Options'         -- Fast-RCNN options parser
require 'fastrcnn.ROIProcessor'
require 'fastrcnn.Transform'

-- load setup/options functions
fastrcnn.train = require 'fastrcnn.train'
fastrcnn.test = require 'fastrcnn.test'
fastrcnn.utils = require 'fastrcnn.utils'
fastrcnn.visualize_detections = require 'fastrcnn.visualize'
]]

paths.dofile('modules/init.lua')         -- bbox modules for train/test
paths.dofile('BatchROISampler.lua') -- data loader/generator
paths.dofile('ImageDetector.lua')   -- single image detector/tester
paths.dofile('Tester.lua')          -- dataset tester
paths.dofile('Options.lua')         -- Fast-RCNN options parser
paths.dofile('ROIProcessor.lua')
paths.dofile('Transform.lua')

-- load setup/options functions
fastrcnn.train = paths.dofile('train.lua')
fastrcnn.test = paths.dofile('test.lua')
fastrcnn.utils = paths.dofile('utils/init.lua')
fastrcnn.visualize_detections = paths.dofile('visualize.lua')

return fastrcnn