(ns dl4clj.clustering.vptree.tree
  (:import [org.deeplearning4j.clustering.vptree VPTree])
  (:require [dl4clj.utils :refer [contains-many?]]))

(defn new-tree
  [& {:keys [item similarity-fn invert? data-points distances]
      :as opts}]
  (assert (or (contains? opts :item)
              (contains? opts :data-points))
          "you must supply data to create the tree")
  (cond (contains-many? opts :data-points :distances :similarity-fn :invert?)
        (VPTree. data-points distances similarity-fn invert?)
        (contains-many? opts :data-points :distances :similarity-fn)
        (VPTree. data-points distances similarity-fn)
        (contains-many? opts :data-points :similarity-fn :invert?)
        (VPTree. data-points similarity-fn invert?)
        (contains-many? opts :item :similarity-fn :invert?)
        (VPTree. item similarity-fn invert?)
        (contains-many? opts :data-points :distances)
        (VPTree. data-points distances)
        (contains-many? opts :data-points :similarity-fn)
        (VPTree. data-points similarity-fn)
        (contains-many? opts :item :similarity-fn)
        (VPTree. item similarity-fn)
        (contains? opts :data-points)
        (VPTree. data-points)
        (contains? opts :item)
        (VPTree. item)
        :else
        (assert false "you must supply data to create the tree and possibly some other params")))

(defn build-from-data
  [data]
  (.buildFromData data))

(defn get-distances
  [tree]
  (.getDistances tree))

(defn get-items
  [tree]
  (.getItems tree))

(defn search!
  [& {:keys [tree target k results distances node priority-q-heap-item]
      :as opts}]
  (assert (or (contains-many? opts :tree :target :k :results :distances)
              (contains-many? opts :tree :node :target :k :priority-q-heap-item))
          "you must supply either combination of values")
  (if (contains? opts :priority-q-heap-item)
    (doto tree (.search node target k priority-q-heap-item))
    (doto tree (.search target k results distances))))

(defn set-distances!
  [& {:keys [tree distances]}]
  (doto tree (.setDistances distances)))

(defn set-items!
  [& {:keys [tree items]}]
  (doto tree (.setItems items)))
