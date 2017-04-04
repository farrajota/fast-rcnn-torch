--[[
    Image/bounding box transformations.
]]


require 'image'

if not fastrcnn then fastrcnn = {} end

------------------------------------------------------------------------------------------------------------

local Transform = torch.class('fastrcnn.Transform')

function Transform:__init(modelParameters, opt, mode)
    assert(opt)
    assert(modelParameters)
    assert(mode)

    -- model
    self.colourspace = modelParameters.colourspace
    self.pixel_scale = modelParameters.pixel_scale
    self.meanstd = modelParameters.meanstd

    -- frcnn options
    self.hflip_prob = (mode=='train' and opt.frcnn_hflip) or 0
    self.scale = (mode=='train' and opt.frcnn_scales) or (mode=='test' and opt.frcnn_test_scales) or 600
    self.max_size = (mode=='train' and opt.frcnn_max_size) or (mode=='test' and opt.frcnn_test_max_size) or 1000

    self.interpolation = 'bicubic'
end

------------------------------------------------------------------------------------------------------------

function Transform:SetColourSpace(input)
    local opts = {
        rgb = function(input) return input end,
        bgr = function(input)
            if input:dim() == 3 then
                return input:index(1, torch.LongTensor{3,2,1})
            elseif input:dim() == 4 then
                return input:index(2, torch.LongTensor{3,2,1})
            else
                error('Input image must be a 3D or 4D Tensor.')
            end
          end,
          yuv = image.rgb2yuv,
          lab = image.rgb2lab,
          hsl = image.rgb2hsl,
          hsv = image.rgb2hsv,
    }
    local convertFn = opts[self.colourspace]
    assert(convertFn, 'Undefined input colour space: ' .. self.colourspace)
    return convertFn(input)
end

------------------------------------------------------------------------------------------------------------

function Transform:ScaleLimit(input)
    local interpolation = self.interpolation or 'bicubic'
    local iW, iH = input:size(3), input:size(2) -- image size
    -- determine scale wrt to the min edge size
    local im_size_min = math.min(iH, iW);
    local im_size_max = math.max(iH, iW);
    local scale = self.scale / im_size_min
    -- Prevent the biggest axis from being more than MAX_SIZE
    if math.floor(scale*im_size_max+0.5) > self.max_size then
        scale = self.max_size / im_size_max
    end
    return image.scale(input, math.floor(iW*scale+0.5), math.floor(iH*scale+0.5), interpolation), scale
end

------------------------------------------------------------------------------------------------------------

function Transform:NormalizePixelLimit(input)
    local min = input:min()
    local max = input:max()
    return input:add(-min):div(max-min)
end

------------------------------------------------------------------------------------------------------------

function Transform:SetPixelScale(input)
    return input:mul(self.pixel_scale)
end

------------------------------------------------------------------------------------------------------------

function Transform:HorizontalFlip(input)
    local is_flipped = false
    if torch.uniform() < self.hflip_prob then
        input = image.hflip(input)
        is_flipped = true
    end
    return input, is_flipped
end

------------------------------------------------------------------------------------------------------------

function Transform:ColourNormalize(input)
    for i=1,input:size(1) do
        if self.meanstd.mean then input[i]:add(-self.meanstd.mean[i]) end
        if self.meanstd.std then input[i]:div(self.meanstd.std[i]) end
    end
    return input
end

------------------------------------------------------------------------------------------------------------

function Transform:ColourJitter()
    --TODO
end

------------------------------------------------------------------------------------------------------------

function Transform:Rotate()
    --TODO
end

------------------------------------------------------------------------------------------------------------

function Transform:image(im)
    local out = im:clone()
    local is_flipped, scale

    -- colourspace convertion
    out = self:SetColourSpace(out)
    -- pixel scale
    out = self:SetPixelScale(out)
    -- flip
    out, is_flipped = self:HorizontalFlip(out)
    -- mean/std norm
    out = self:ColourNormalize(out)
    -- scale
    out, scale = self:ScaleLimit(out)

    collectgarbage()

    return out, scale, {out:size(2), out:size(3)}, is_flipped
end
