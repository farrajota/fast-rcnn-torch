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

paths.dofile('code/ROIPooling.lua') -- this package requies the use of this layer for best accu performance

fastrcnn = {}

-- load image detector class
paths.dofile('code/Detector.lua') 

-- load setup/options functions
fastrcnn.train = paths.dofile('code/train.lua')
fastrcnn.test = paths.dofile('code/test.lua')
fastrcnn.options = paths.dofile('code/options.lua')
fastrcnn.utils = {
    visualize_detections = paths.dofile('code/util/visualize.lua'),
    model = {
        setup = paths.dofile('code/model.lua'),
        store = paths.dofile('code/util/store.lua')
    },
    data = {
        loadmatlab = paths.dofile('code/util/loadmatlab.lua')
    }
}

return fastrcnn