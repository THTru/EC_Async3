class RPGView {
  static final int RADIUS = 8; 
  static final int SIZE = RADIUS * 2 + 1;
  float MARGIN_X = (width - RPGTile.WIDTH * SIZE) / 2;
  float MARGIN_Y = (height - RPGTile.HEIGHT * SIZE) / 2;

  RPGAvatar avatar;
  RPGMap map;

  RPGView(RPGAvatar avatar, RPGMap map) {
    this.avatar = avatar;
    this.map = map;
  }

  void moveAvatar(int direction) {
    int x = avatar.x;
    int y = avatar.y;
    switch(direction) {
    case LEFT:
      x--;
      break;
    case UP:
      y--;
      break;
    case DOWN: 
      y++;
      break;
    case RIGHT:
      x++;
      break;
    }
    if (x < 0 || x > this.map.map[0].length() || y < 0 || y > this.map.map.length)
      return;
    RPGTile tile = map.getTile(x, y);
    if (tile==RPGTile.MOUNTAIN || tile==RPGTile.RIVER || tile==RPGTile.ROCK) return; 
    avatar.x = x;
    avatar.y = y;
  }

  void draw() {
    int x_min = avatar.x - RADIUS;
    int y_min = avatar.y - RADIUS;
    for (int x = 0; x < SIZE; x++) {
      for (int y = 0; y < SIZE; y++) {
        RPGTile tile = map.getTile(x + x_min, y + y_min);
        PImage image = loadImage(tile.image);
        image.resize(RPGTile.WIDTH, RPGTile.HEIGHT);
        image(image, MARGIN_X + RPGTile.WIDTH * x, MARGIN_Y + RPGTile.HEIGHT * y);
      }
    }
    image(avatar.image, MARGIN_X + RADIUS * RPGTile.WIDTH, MARGIN_Y +RADIUS * RPGTile.HEIGHT);
  }
}
