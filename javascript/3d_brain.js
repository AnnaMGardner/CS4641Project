var loadedModel = taccgl.objFile().read( '/3d_brain.obj');
taccgl.actor("braindiv", loadedModel.scene()) . rotateMiddle(1,1,1) . duration(10) .start();
