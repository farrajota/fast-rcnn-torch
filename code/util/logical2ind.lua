--[[
    
]]

local function logical2ind(logical)
  if logical:numel() == 0 then
    return torch.LongTensor()
  end
  return torch.range(1,logical:numel())[logical:gt(0)]:long()
end

-----------------------------------------------

return logical2ind