package = "fastrcnn"

version = "scm-1"

source = {
    url = "git://github.com/farrajota/fast-rcnn-torch.git",
    tag = "master"
 }

description = {
    summary = "Fast-RCNN package for Torch7.",
    detailed = [[
       Fast-RCNN implementation for Torch7. This package allows to easily train, test and implement an FRCNN object detector in Lua.
    ]],
    homepage = "https://github.com/farrajota/fast-rcnn-torch",
    license = "MIT",
    maintainer = "Farrajota"
 }

dependencies = {
    "torch >= 7.0",
    "cudnn >= scm-1",
    "tds >= scm-1",
    "matio >= scm-1",
    "torchnet >= scm-1",
    "inn >= 1.0-0"
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