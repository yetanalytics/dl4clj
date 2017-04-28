(ns dl4clj.clustering.sptree.heap-item
  (:import [org.deeplearning4j.clustering.sptree HeapItem]))

(defn new-heap-item
  [idx distance]
  (HeapItem. idx distance))

(defn compare-to
  [item-1 item-2]
  (.compareTo item-1 item-2))

(defn get-distance
  [item]
  (.getDistance item))

(defn get-idx
  [item]
  (.getIndex item))

(defn set-distance!
  [item distance]
  (doto item (.setDistance distance)))

(defn set-idx!
  [item idx]
  (doto item (.setIndex idx)))
