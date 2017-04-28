(ns dl4clj.clustering.sptree.cell
  (:import [org.deeplearning4j.clustering.sptree Cell])
  (:require [dl4clj.utils :refer [contains-many?]]))

(defn new-cell
  [dimension]
  (Cell. dimension))

(defn contains-point?
  [cell point]
  (.contains cell point))

(defn corner
  [& {:keys [cell dimension]
      :as opts}]
  (assert (contains? opts :cell) "you must provide a cell to find the corner")
  (if (contains? opts :dimension)
    (.corner cell dimension)
    (.corner cell)))

(defn set-corner!
  [& {:keys [cell corner-array dimension corner]
      :as opts}]
  (assert (contains? opts :cell) "you must provide a cell to set its corner")
  (if (contains-many? opts :dimension :corner)
    (doto cell (.setCorner dimension corner))
    (doto cell (.setCorner corner-array))))

(defn set-width!
  [& {:keys [cell width-array dimension width]
      :as opts}]
  (assert (contains? opts :cell) "you must provide a cell to set its width")
  (if (contains-many? opts :dimension :width)
    (doto cell (.setWidth dimension width))
    (doto cell (.setWidth width-array))))

(defn width
  [& {:keys [cell width]
      :as opts}]
  (assert (contains? opts :cell) "you must provide a cell to get the width")
  (if (contains? opts :width)
    (.width cell width)
    (.width cell)))
