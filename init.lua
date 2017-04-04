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

require 'modules'         -- bbox modules for train/test
require 'BatchROISampler' -- data loader/generator
require 'ImageDetector'   -- single image detector/tester
require 'Tester'          -- dataset tester
require 'Options'         -- Fast-RCNN options parser
require 'ROIProcessor'
require 'Transform'

-- load setup/options functions
fastrcnn.train = require 'train'
fastrcnn.test = require 'test'
fastrcnn.utils = require 'utils'
fastrcnn.visualize_detections = require 'utils.visualize'

return fastrcnn