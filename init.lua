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
fastrcnn.visualize_detections = require 'fastrcnn.utils.visualize'

return fastrcnn