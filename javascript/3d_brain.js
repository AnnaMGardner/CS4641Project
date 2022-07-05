var loadedModel = taccgl.objFile().read( '/objtest/roundcube/roundcube1.obj');
taccgl.actor("braindiv", loadedModel.scene()) . rotateMiddle(1,1,1) . duration(10) .start();
