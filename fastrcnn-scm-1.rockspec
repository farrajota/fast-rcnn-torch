package = "fastrcnn"

version = "scm-1"

source = {
    url = "git://github.com/farrajota/fast-rcnn.git",
    tag = "master"
 }
 
description = {
    summary = "Fast-RCNN code for Torch7.",
    detailed = [[
       Fast-RCNN implementation for Torch7. This package allows to train, test and implement an object detector.
    ]],
    homepage = "https://github.com/farrajota/fast-rcnn",
    license = "BSD",
    maintainer = "Farrajota"
 }

dependencies = {
    "lua ~> 5.1",
    "torch >= 7.0",
    "tds >= scm-1",
    "matio >= scm-1",
    "torchnet >= scm-1"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build;
cd build;
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"; 
$(MAKE)
   ]],
   install_command = "cd build && $(MAKE) install"
}