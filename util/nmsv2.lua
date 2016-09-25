local ffi = require 'ffi'
ffi.cdef[[
void bbox_vote(THFloatTensor *res, THFloatTensor *nms_boxes, THFloatTensor *scored_boxes, float threshold);
void NMS(THFloatTensor *keep, THFloatTensor *scored_boxes, float overlap);
]]

local ok, C = pcall(ffi.load, paths.concat(projectDir,'code','util','libnms.so'))
--if not ok then
--   os.execute('make')
--   ok, C = pcall(ffi.load, './libnms.so')
--   assert(ok, 'run make and check what is wrong')
--end


local function nms(boxes, overlap)
   local keep = torch.FloatTensor()
   C.NMS(keep:cdata(), boxes:cdata(), overlap)
   return keep
end

return nms