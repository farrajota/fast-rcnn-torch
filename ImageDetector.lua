--[[
    Object detector class. Receives images and region proposals as inputs and outputs scores and bounding boxes.
]]


local transform = paths.dofile('transform.lua')
local utils = paths.dofile('utils/init.lua')

------------------------------------------------------------------------------------------------

local detector = torch.class('fastrcnn.ImageDetector')

function detector:__init(model, opt, model_parameters)
  
  assert(model)
  assert(opt)
  assert(model_parameters)
  
  self.model = model
  self.opt = opt
  self.opt.model_params = model_parameters
  
  -- setup data augment/normalization function
  self.transformDataFn = transform('test', self.opt)
  
  -- softmax function
  self._softmax = nn.SoftMax():float()
  
  -- image detection storages (this avoids constant allocs while evaluation images)  
  self.img_detection = torch.FloatTensor()
  self.roi_boxes_detection = torch.FloatTensor()
  
  -- set buffers to GPU or CPU
  if self.opt.GPU >= 1 then
      self.model:cuda()
      self._softmax = self._softmax:cuda()
      self.img_detection = self.img_detection:cuda()
      self.roi_boxes_detection = self.roi_boxes_detection:cuda()
  end
  
  -- set model to evaluation mode
  self.model:evaluate()
end


function detector:detect(im, boxes) -- Detect objects in an image
  -- preprocess image
  local img_transf, boxes_transf = self.transformDataFn(im, boxes)
  
  -- prepare image + roi boxes data
  local img_size = img_transf:size()
  local img = self.img_detection:resize(1, img_size[1], img_size[2], img_size[3])
  img[1]:copy(img_transf)
  local roi_boxes = self.roi_boxes_detection:resize(boxes_transf:size(1), 5)
  roi_boxes[{{},{1}}]:fill(1)
  roi_boxes[{{},{2,5}}]:copy(boxes_transf)
  
  -- forward pass through the network
  local outputs = self.model:forward({img, roi_boxes})
  
  -- fetch data from the network
  local scores, predicted_boxes
  if type(outputs)=='table' then
      scores =  self._softmax:forward(outputs[1])
      predicted_boxes = utils.box_transform.TransformInv(boxes, outputs[2]:float(), im:size())
     -- predicted_boxes = boxes:repeatTensor(1,21)
  else
      scores =  self._softmax:forward(outputs)
      predicted_boxes = boxes
  end
  
  return scores:float(), predicted_boxes:float()
end