--[[
    Table utility functions.
]]


local tnt = require 'torchnet'

------------------------------------------------------------------------------------------------------------

local function convert_tds_to_table(input)
    assert(input)
    local out = {}
    for k, v in pairs(input) do
        out[k] = v
    end
    return out
end

------------------------------------------------------------------------------------------------------------

local function tds_to_table(input)
  assert(input)

  local type_input = type(input)
  local ttype_input = torch.type(input)

  if type_input == 'table' then
      return input
  elseif type_input == 'cdata' then
      if string.lower(ttype_input) == 'tds.hash' or string.lower(ttype_input) == 'tds.hash' then
          return convert_tds_to_table(input)
      else
          error('Input must be either a tds.hash, tds.vec or a table: ' .. ttype_input)
      end
  else
      error('Input must be either a tds.hash, tds.vec or a table: ' .. ttype_input)
  end
end

------------------------------------------------------------------------------------------------------------

-- source: https://github.com/facebookresearch/multipathnet/blob/d677e798fcd886215b1207ae1717e2e001926b9c/utils.lua#L374
local function recursiveCopy(t1,t2)
   if torch.type(t2) == 'table' then
      t1 = (torch.type(t1) == 'table') and t1 or {t1}
      for key,_ in pairs(t2) do
         t1[key], t2[key] = recursiveCopy(t1[key], t2[key])
      end
   elseif torch.isTensor(t2) then
      t1 = torch.isTensor(t1) and t1 or t2.new()
      t1:resize(t2:size()):copy(t2)
   elseif torch.type(t2) == 'number' then
      t1 = t2
   else
      error("expecting nested tensors or tables. Got "..
      torch.type(t1).." and "..torch.type(t2).." instead")
   end
   return t1, t2
end

------------------------------------------------------------------------------------------------------------

-- source: https://github.com/facebookresearch/multipathnet/blob/d677e798fcd886215b1207ae1717e2e001926b9c/utils.lua#L394
local function recursiveCast(dst, src, type)
   if #dst == 0 then
      tnt.utils.table.copy(dst, nn.utils.recursiveType(src, type))
   end
   recursiveCopy(dst, src)
end

------------------------------------------------------------------------------------------------------------

local function ConcatTables(tableA, tableB)
    local tableOut = {}
    for i=1, #tableA do
        table.insert(tableOut, tableA[i])
    end
    for i=1, #tableB do
        table.insert(tableOut, tableB[i])
    end
    return tableOut
end

------------------------------------------------------------------------------------------------------------

local function joinTable(input,dim)
    local size = torch.LongStorage()
    local is_ok = false
    for i=1,#input do
        local currentOutput = input[i]
        if currentOutput:numel() > 0 then
            if not is_ok then
                size:resize(currentOutput:dim()):copy(currentOutput:size())
                is_ok = true
            else
                size[dim] = size[dim] + currentOutput:size(dim)
            end
        end
    end
    local output = input[1].new():resize(size)
    local offset = 1
    for i=1,#input do
        local currentOutput = input[i]
        if currentOutput:numel() > 0 then
            output:narrow(dim, offset, currentOutput:size(dim)):copy(currentOutput)
            offset = offset + currentOutput:size(dim)
        end
    end
    return output
end

------------------------------------------------------------------------------------------------------------

return {
    tds_to_table = tds_to_table,
    recursiveCopy = recursiveCopy,
    recursiveCast = recursiveCast,
    concatTables = ConcatTables,
    joinTable = joinTable,
}