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


paths.dofile('ROIPooling.lua') -- this package requies the use of this layer for best accu performance

fastrcnn = {}

-- load image detector class
paths.dofile('ImageDetector.lua')
-- load image tester class
paths.dofile('Tester.lua')

-- load setup/options functions
fastrcnn.train = paths.dofile('train.lua')
fastrcnn.test = paths.dofile('test.lua')
fastrcnn.utils = paths.dofile('utils/init.lua') 

return fastrcnn