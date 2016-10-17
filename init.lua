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
fastrcnn.train = paths.dofile('train.lua')
fastrcnn.test = paths.dofile('test.lua')
fastrcnn.utils = paths.dofile('utils/init.lua') 
--[[{
    visualize_detections = paths.dofile('code/util/visualize.lua'),
    model = {
        store = paths.dofile('code/util/store.lua')
    },
    data = {
        loadmatlab = paths.dofile('code/util/loadmatlab.lua')
    }
}
]]

return fastrcnn