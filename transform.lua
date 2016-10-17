--[[
    Transform data function.
]]

local ti = paths.dofile('transform_image.lua') -- image transformation functions
local tb = paths.dofile('transform_box.lua') -- box transformation functions

--------------------------------------------------------------------------------

local function transform(mode, opts)

 -- data augment options
  local max_size = (mode == 'train' and opts.frcnn_max_size) or (mode == 'test' and opts.frcnn_test_max_size)
  local scale = (mode == 'train' and opts.frcnn_scales[1]) or (mode == 'test' and opts.frcnn_test_scales[1])
  local hflip = (mode == 'train' and opts.frcnn_hflip) or (mode == 'test' and math.huge)
  local rotate = (mode == 'train' and opts.frcnn_rotate) or (mode == 'test' and 0) 
  local jitter = (mode == 'train' and opts.frcnn_jitter) or (mode == 'test' and 0)
  
  -- model parameters
  local meanstd = {mean = opts.model_params.mean, std = opts.model_params.std}
  local colourspace = opts.model_params.colourspace
  local pixelscale = opts.model_params.pixel_scale
  
  -- roi proposals parameters
  local roi_meanstd = opts.roi_meanstd

  -- data augment/normalization function
  return function (img, boxes)
    
    local out = img:clone()
    local bbox = boxes:clone()
    
    -- colourspace convert
    if colourspace then
        out = ti.ColorSpace(colourspace)(out) 
    end
  
    -- pixel range scale
    if pixelscale then
        out = ti.Mul(pixelscale)(out)
    end
    
    -- scale
    out, img_scale = ti.ScaleLimit(scale, max_size)(out) 
    bbox = tb.Scale(img_scale)(bbox)
    
    -- horizontal flip
    if torch.uniform() > hflip then
        out = ti.HorizontalFlip(math.huge)(out)
        bbox = tb.HorizontalFlip(out:size(3))(bbox)
    end
    
    -- rotate
--    out = ti.HorizontalFlip(math.huge)(out) 
--    bbox = tb.HorizontalFlip()(bbox)
    
    -- jitter
    --out = ti.Jitter(math.huge)(out)
    --bbox = tb.Jitter()(bbox)
    
    -- hsv augment
    
    -- Subtract mean/std
    if meanstd then
        out = ti.ColorNormalize(meanstd)(out)
    end
    
    return out, bbox
  end
  
end

--------------------------------------------------------------------------------

return transform