(ns dl4clj.clustering.sptree.data-point
  (:import [org.deeplearning4j.clustering.sptree DataPoint])
  (:require [dl4clj.utils :refer [contains-many?]]))

(defn new-data-point
  [{:keys [idx point invert? fn-name]
    :as opts}]
  (assert (contains-many? opts :idx :point) "you must provide an index and an array describing the point")
  (cond (contains-many? opts :fn-name :invert?)
        (DataPoint. idx point fn-name invert?)
        (contains? opts :fn-name)
        (DataPoint. idx point fn-name)
        (contains? opts :invert?)
        (DataPoint. idx point invert?)
        :else
        (DataPoint. idx point)))

(defn get-euclidean-distance
  [& {:keys [point-1 point-2]}]
  (.distance point-1 point-2))

(defn get-d
  [data-point]
  (.getD data-point))

(defn get-idx
  [data-point]
  (.getIndex data-point))

(defn get-point
  [data-point]
  (.getPoint data-point))

(defn set-d!
  [& {:keys [data-point d]}]
  (doto data-point (.setD d)))

(defn set-idx!
  [& {:keys [data-point idx]}]
  (doto data-point (.setIndex idx)))

(defn set-point!
  [& {:keys [data-point point]}]
  (doto data-point (.setPoint point)))
