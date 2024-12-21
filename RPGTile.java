import processing.core.*;

class RPGTile {
  static final int WIDTH = 64;
  static final int HEIGHT = 64;
  String image;

  RPGTile(String filename) {
    this.image = filename;
  }

  static RPGTile getTile(String type) {

    switch(type) {
    case "0":
      return MOUNTAIN;
    case "1":
      return RIVER;
    case "2":
      return GRASS;
    case "3":
      return ROCK;
    case "4":
      return RIVERROCK;
    }
    return EMPTY;
  }

  static RPGTile MOUNTAIN = new RPGTile("mountain.png");
  static RPGTile RIVER = new RPGTile("river.png");
  static RPGTile GRASS = new RPGTile("grass.png");
  static RPGTile ROCK = new RPGTile("rock.png");
  static RPGTile RIVERROCK = new RPGTile("riverstone.png");
  static RPGTile EMPTY = new RPGTile("empty.png");
}
