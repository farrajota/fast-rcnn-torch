--[[
     Non-maximum suppression.
]]

local ffi = require 'ffi'
ffi.cdef[[
void bbox_vote(THFloatTensor *res, THFloatTensor *nms_boxes, THFloatTensor *scored_boxes, float threshold);
void NMS(THFloatTensor *keep, THFloatTensor *scored_boxes, float overlap);
]]

--local ok, C = pcall(ffi.load, paths.concat(projectDir,'code','util','libnms.so'))
local ok, C = pcall(ffi.load, package.searchpath('libfastrcnn', package.cpath))
assert(ok, 'Installation went wrong when compiling the C code.')

local function nms(boxes, overlap)
   local keep = torch.FloatTensor()
   C.NMS(keep:cdata(), boxes:cdata(), overlap)
   return keep
end

return nms