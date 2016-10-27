--[[
    Bounding Box transformation library. It contains common  transformations like rotation, flip, jitter, etc.
]]

require 'image'

local M = {}

-- flip
function M.HorizontalFlip()
   return function(input, width)
      local out = input:clone()
      if out:dim() > 1 then
          local tmp = out[{{},{1}}]:clone()
          out[{{},{1}}] = -(out[{{},{3}}]-width) + 1
          out[{{},{3}}] = -(tmp-width) + 1
      else
          local tmp = out[1]
          out[1] = width-out[3] +1
          out[3] = width-tmp +1
      end
      return out
   end
end

-- rotate
function M.Rotate(deg)
   return function(input)
      local out = input:clone()
      
      return out
   end
end

-- jitters the bbox coordinates
function M.Jitter(jitW, jitH)
   return function(input)
      local out = input:clone():fill(0)
      local iW, iH = input:size(3), input:size(2)
      local jW = torch.random(0,jitW) * (torch.uniform() - 0.5)
      local jH = torch.random(0,jitH) * (torch.uniform() - 0.5)
      out[{{},{math.max(1, -jH), math.min(iH, iH-jH)}, {math.max(1, -jW), math.min(iW, iW-jW)}}]:copy(
        input[{{},{math.max(1,jH), math.min(iH-jH, iH)},{math.max(1,jW), math.min(iW-jW, iW)}}])
      return out
   end
end

-- scale
function M.Scale_old(scales)
   return function(input)
      -- source: https://github.com/mahyarnajibi/fast-rcnn-torch/blob/aaa0a33805a6ca761281bde7994900127d738daa/ROI/ROI.lua#L223-L238
      local rois = torch.FloatTensor()
      local total_bboxes = 0
      local cumul_bboxes = {0}
      for i=1,#scales do
        total_bboxes = total_bboxes + input[i]:size(1)
        table.insert(cumul_bboxes, total_bboxes)
      end
      rois:resize(total_bboxes,5)
      for i=1,#scales do
        local idx = {cumul_bboxes[i]+1, cumul_bboxes[i+1]}
        rois[{idx,1}]:fill(i)
        rois[{idx,{2,5}}]:copy(input[i]):add(-1):mul(scales[i]):add(1)
      end
      
      return rois
   end
end

-- scale
function M.Scale(scales)
   return function(input)
      -- source: https://github.com/mahyarnajibi/fast-rcnn-torch/blob/aaa0a33805a6ca761281bde7994900127d738daa/ROI/ROI.lua#L223-L238
      local rois = input:clone()
      rois:add(-1):mul(scales):add(1)
      return rois
   end
end

-- scale
function M.Scale_altered(max_size)
   return function(input, scales)
      -- source: https://github.com/mahyarnajibi/fast-rcnn-torch/blob/aaa0a33805a6ca761281bde7994900127d738daa/ROI/ROI.lua#L223-L238
      return input:clone():add(-1):mul(scales):add(1):clamp(0,max_size)
   end
end

function M.Compose(transforms)
   return function(input)
      for _, transform in ipairs(transforms) do
         input = transform(input)
      end
      return input
   end
end

function M.collectgarbage()
    return function(input)
        collectgarbage(); collectgarbage();
        return input
    end
end

return M