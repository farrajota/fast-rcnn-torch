--[[
    Image/bounding box transformations.
]]


local ti = paths.dofile('transform_image.lua') -- image transformation functions
local tb = paths.dofile('transform_box.lua') -- box transformation functions

--------------------------------------------------------------------------------------------------

local Transform = torch.class('fastrcnn.Transform')

function Transform:__init(opt, mode)
    assert(opt)
    assert(mode)
    
    -- model
    self.colourspace = opt.colourspace
    self.pixel_scale = opt.pixel_scale
    self.meanstd = opt.meanstd
    
    -- frcnn options
    self.flip = (mode=='train' and opt.frcnn_hflip) or 0
    self.scale = (mode=='train' and opt.frcnn_scales[1]) or (mode=='test' and opt.frcnn_test_scales[1]) or 600
    self.max_size = (mode=='train' and opt.frcnn_max_size) or (mode=='test' and opt.frcnn_test_max_size) or 1000
    
    -- image transform functions
    self.Mul = ti.Mul(self.pixel_scale)
    self.ScaleLimit = ti.ScaleLimit(self.scale, self.max_size, 'bicubic')
    self.ColourNormalize = ti.ColorNormalize(self.meanstd)
    self.HorizontalFlip = ti.HorizontalFlip(self.flip)
    self.Colourspace = ti.ColorSpace(self.colourspace)
    self.PixelLimit = ti.PixelLimit()
    
    -- bbox transform functions
    self.bbScale = tb.Scale()
    self.bbHorizontalFlip = tb.HorizontalFlip()
end


function Transform:image(im)
    -- colourspace transform
    local im_transf = self.Colourspace(im)
    local is_flipped, scale
    
    -- scale
    im_transf, scale = self.ScaleLimit(im_transf)
    -- normalize values between 0 and 1 after scaling
    im_transf = self.PixelLimit(im_transf)
    -- pixel scale
    im_transf = self.Mul(im_transf)
    -- flip
    im_transf, is_flipped = self.HorizontalFlip(im_transf)
    -- mean/std norm
    im_transf = self.ColourNormalize(im_transf)
    
    collectgarbage()
    
    return im_transf, scale, {im_transf:size(2), im_transf:size(3)}, is_flipped
end


function Transform:bbox(bboxes, im_scale, im_size, is_flipped)
    -- scale
    local bboxes_transf = self.bbScale(bboxes, im_scale)
    -- horizontal flip
    if is_flipped then
        bboxes_transf = self.bbHorizontalFlip(bboxes_transf, im_size[2])
    end
    
    collectgarbage()
    
    return bboxes_transf
end
