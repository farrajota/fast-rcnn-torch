--[[
    Bounding box transformations.
    
    Source: 
      https://github.com/mahyarnajibi/fast-rcnn-torch/blob/aaa0a33805a6ca761281bde7994900127d738daa/ROI/ROI.lua#L9
      https://github.com/ShaoqingRen/faster_rcnn/blob/d76c23cc09b55eb0472d4707f4f91dd83b35958a/functions/fast_rcnn/fast_rcnn_bbox_transform_inv.m)
]]


local function ClampValues(boxes, im_size)  
  -- adapted from https://github.com/mahyarnajibi/fast-rcnn-torch/blob/aaa0a33805a6ca761281bde7994900127d738daa/ROI/ROI.lua#L197  
  -- clamp values between 1 and max_size
  boxes = boxes:clamp(1, math.max(im_size[2], im_size[3]))
  
  -- clamp the other edge values if it's smaller than max_size
  if im_size[2] > im_size[3] then
      local x2_inds = torch.range(3, boxes:size(2), 4):long()
      local x2 = boxes:index(2, x2_inds)
      x2[x2:gt(im_size[3])] = im_size[3]
      boxes:indexCopy(2, x2_inds, x2)
  elseif im_size[2] < im_size[3] then
      local y2_inds = torch.range(4, boxes:size(2), 4):long()
      local y2 = boxes:index(2, y2_inds)
      y2[y2:gt(im_size[2])] = im_size[2]
      boxes:indexCopy(2, y2_inds, y2)
  end

  return boxes
end

-------------------------------------------------------------------------------------------------------------
local function box_transform(bboxes,targets)
--[[
    Description:
        Transforms bbox coordinates from [x1,y1,x2,y2] to [x,y,w,h] and normalizes them wrt the ground-truth boxes.
    
    Arguments
        roi_boxes
        gt_boxes
        
    Return values
        reg_box
       
    Example
        
]]
    
  -- The function encodes the delta in the bounding box regression
  local bb_widths = bboxes[{{},{3}}] - bboxes[{{},{1}}] + 1e-14
  local bb_heights = bboxes[{{},{4}}] - bboxes[{{},{2}}] + 1e-14
  local bb_ctr_x = bboxes[{{},{1}}] + bb_widths * 0.5
  local bb_ctr_y = bboxes[{{},{2}}] + bb_heights * 0.5

  local target_widths = targets[{{},{3}}] - targets[{{},{1}}] + 1e-4
  local target_heights = targets[{{},{4}}] - targets[{{},{2}}] + 1e-4
  local target_ctr_x = targets[{{},{1}}] + target_widths * 0.5
  local target_ctr_y = targets[{{},{2}}] + target_heights * 0.5

  local targets = torch.FloatTensor(bboxes:size(1),4)
  targets[{{},{1}}] = torch.cdiv((target_ctr_x - bb_ctr_x), bb_widths)
  targets[{{},{2}}] = torch.cdiv((target_ctr_y - bb_ctr_y), bb_heights)
  targets[{{},{3}}] = torch.log(torch.cdiv(target_widths,bb_widths))
  targets[{{},{4}}] = torch.log(torch.cdiv(target_heights,bb_heights))

  return targets
end

-------------------------------------------------------------------------------------------------------------

local function box_transform_inv(boxes,box_deltas,im_size)
--[[
    Description:
        Apply the inverse transform to the bbox regression coordinates from [x,y,w,h] to [x1,y1,x2,y2] and normalize them wrt the box deltas. (Adapted from https://github.com/mahyarnajibi/fast-rcnn-torch/blob/aaa0a33805a6ca761281bde7994900127d738daa/ROI/ROI.lua#L9 and https://github.com/ShaoqingRen/faster_rcnn/blob/d76c23cc09b55eb0472d4707f4f91dd83b35958a/functions/fast_rcnn/fast_rcnn_bbox_transform_inv.m)
    
    Arguments
        boxes
        box_deltas
        im_size: (channels, height, width)
        
    Return values
        pred_boxes
       
    Example
        
]]

    -- Function to decode the output of the network
  -- Check to see whether boxes are empty or not 
  if boxes:size()[1] == 0 then
    return torch.FloatTensor(0,boxes:size()[2]):zero()
  end

  box_deltas = box_deltas:float()
  local widths = boxes[{{},{3}}]:float() - boxes[{{},{1}}]:float() + 1e-14
  local heights = boxes[{{},{4}}]:float() - boxes[{{},{2}}]:float() + 1e-14
  local centers_x = boxes[{{},{1}}]:float() + widths * 0.5
  local centers_y = boxes[{{},{2}}]:float() + heights * 0.5

  local x_inds = torch.range(1,box_deltas:size()[2],4):long()
  local y_inds = torch.range(2,box_deltas:size()[2],4):long()
  local w_inds = torch.range(3,box_deltas:size()[2],4):long()
  local h_inds = torch.range(4,box_deltas:size()[2],4):long()

  local dx = box_deltas:index(2,x_inds)
  local dy = box_deltas:index(2,y_inds)
  local dw = box_deltas:index(2,w_inds)
  local dh = box_deltas:index(2,h_inds)


  local predicted_center_x = dx:cmul(widths:expand(dx:size())) + centers_x:expand(dx:size())
  local predicted_center_y = dy:cmul(heights:expand(dy:size())) + centers_y:expand(dy:size())
  local predicted_w = torch.exp(dw):cmul(widths:expand(dw:size()))
  local predicted_h = torch.exp(dh):cmul(heights:expand(dh:size()))

  local predicted_boxes = torch.FloatTensor(box_deltas:size()):zero()
  local half_w = predicted_w * 0.5
  local half_h = predicted_h * 0.5
  predicted_boxes:indexCopy(2,x_inds,predicted_center_x - half_w)
  predicted_boxes:indexCopy(2,y_inds,predicted_center_y -  half_h)
  predicted_boxes:indexCopy(2,w_inds,predicted_center_x + half_w)
  predicted_boxes:indexCopy(2,h_inds,predicted_center_y + half_h)
  predicted_boxes = ClampValues(predicted_boxes,im_size)

  return predicted_boxes
end

-------------------------------------------------------------------------------------------------------------

return {
    ClampValues = ClampValues,
    Transform = box_transform,
    TransformInv = box_transform_inv
}