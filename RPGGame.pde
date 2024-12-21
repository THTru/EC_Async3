RPGMap map;
RPGView view;
RPGAvatar avatar;

public void settings() {  
  fullScreen();
}

public void setup() {
  map = new RPGMap("default_1.map");
  avatar = new RPGAvatar(map.x_start, map.y_start);
  view = new RPGView(avatar, map);
  frameRate(10);
  background(0);
  view.draw();
}

public void draw() {

  if (keyPressed) {
    view.moveAvatar(keyCode);        
    view.draw();
  }
}
