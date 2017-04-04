--[[
    Object detector class. Receives images and region proposals as inputs and outputs scores and bounding boxes.
]]


local utils = require 'utils'

if not fastrcnn then fastrcnn = {} end

------------------------------------------------------------------------------------------------

local detector = torch.class('fastrcnn.ImageDetector')


function detector:__init(model, modelParameters, opt)

    assert(model)
    assert(modelParameters)
    assert(opt)

    self.model = model

    -- set model to use only one GPU
    utils.model.setDataParallel(self.model, opt.GPU or 1, 1)

    self.scales = opt.frcnn_test_scales
    self.max_size = opt.frcnn_test_max_size

    -- setup data augment/normalization function
    self.Transformer = fastrcnn.Transform(modelParameters, opt, 'test') -- image transformer

    -- softmax function
    self._softmax = nn.SoftMax():float()

    -- image detection storages (this avoids constant allocs while evaluation images)
    self.img_detection = torch.FloatTensor()
    self.roi_boxes_detection = torch.FloatTensor()

    -- set buffers to GPU or CPU
    if opt.GPU >= 1 then
        self.model:cuda()
        self._softmax = self._softmax:cuda()
        self.img_detection = self.img_detection:cuda()
        self.roi_boxes_detection = self.roi_boxes_detection:cuda()
    end

    -- set model to evaluation mode
    self.model:evaluate()
end

------------------------------------------------------------------------------------------------------------

function detector:getImageBoxes(im, boxes)
    -- preprocess image
    local img_transf, scale, _, _ = self.Transformer:image(im)

    -- reshape image + roi boxes buffers and fill them with data
    local img_size = img_transf:size()
    local img = self.img_detection:resize(1, img_size[1], img_size[2], img_size[3])
    img[1]:copy(img_transf)
    local roi_boxes = self.roi_boxes_detection:resize(boxes:size(1), 5)
    roi_boxes[{{},{1}}]:fill(1)
    roi_boxes[{{},{2,5}}]:copy(boxes):add(-1):mul(scale):add(1)--:clamp(1,self.max_size)

    return img, roi_boxes
end

------------------------------------------------------------------------------------------------------------

function detector:detect(im, boxes) -- Detect objects in an image
    assert(im)
    assert(boxes)

    local img, roi_boxes = self:getImageBoxes(im, boxes)

    -- forward pass through the network
    local outputs = self.model:forward({img, roi_boxes})

    -- fetch data from the network
    local scores, predicted_boxes
    if type(outputs)=='table' then
        scores = self._softmax:forward(outputs[1])
        predicted_boxes = outputs[2]:float()
        for i,v in ipairs(predicted_boxes:split(4,2)) do
            utils.box.convertFrom(v,boxes,v)
        end
    else
        scores =  self._softmax:forward(outputs)
        predicted_boxes = boxes:repeatTensor(1,outputs:size(2))
    end

    return scores:float(), predicted_boxes:float()
end