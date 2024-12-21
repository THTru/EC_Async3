class RPGMap {

  String[] map;
  int x_start, y_start;

  RPGMap(String filename) {

    map = loadStrings(filename);
    x_start = map[0].length() / 2;
    y_start = map.length / 2;
  }

  RPGTile getTile(int x, int y) {

    if (y < 0 || y > map.length - 1 ||
      x < 0 || x + 1 > map[y].length() - 1) {
      return RPGTile.EMPTY;
    }
    String type = map[y].substring(x, x + 1);
    return RPGTile.getTile(type);
  }
}
