(ns dl4clj.clustering.quadtree.cell
  (:import [org.deeplearning4j.clustering.quadtree Cell]))

(defn new-cell
  [x y hw hh]
  (Cell. x y hw hh))

(defn contains-point?
  [cell point]
  (.containsPoint cell point))

(defn get-hh
  [cell]
  (.getHh cell))

(defn get-hw
  [cell]
  (.getHw cell))

(defn get-x
  [cell]
  (.getX cell))

(defn get-y
  [cell]
  (.getY cell))

(defn set-hh!
  [cell hh]
  (doto cell (.setHh hh)))

(defn set-hw!
  [cell hw]
  (doto cell (.setHw hw)))

(defn set-x!
  [cell x]
  (doto cell (.setX x)))

(defn set-y!
  [cell y]
  (doto cell (.setY y)))
