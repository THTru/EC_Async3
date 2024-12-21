class RPGAvatar {

  int x, y;
  PImage image;

  RPGAvatar(int x_start, int y_start) {
    x = x_start;
    y = y_start;
    image = loadImage("avatar.png");
    image.resize(RPGTile.WIDTH, RPGTile.HEIGHT);
  }
}
