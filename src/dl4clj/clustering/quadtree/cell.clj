(ns dl4clj.clustering.quadtree.cell
  (:import [org.deeplearning4j.clustering.quadtree Cell]))

(defn new-cell
  [& {:keys [x y hw hh]}]
  (Cell. x y hw hh))

(defn contains-point?
  [& {:keys [cell point]}]
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
  [& {:keys [cell hh]}]
  (doto cell (.setHh hh)))

(defn set-hw!
  [& {:keys [cell hw]}]
  (doto cell (.setHw hw)))

(defn set-x!
  [& {:keys [cell x]}]
  (doto cell (.setX x)))

(defn set-y!
  [& {:keys [cell y]}]
  (doto cell (.setY y)))
