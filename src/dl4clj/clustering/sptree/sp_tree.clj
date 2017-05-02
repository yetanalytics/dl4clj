(ns dl4clj.clustering.sptree.sp-tree
  (:import [org.deeplearning4j.clustering.sptree SpTree])
  (:require [dl4clj.utils :refer [contains-many?]]))

(defn new-sptree
  [& {:keys [data parent indices similarity-fn corner width]
      :as opts}]
  (assert (contains? opts :data) "you must provide data to create the tree with")
  (cond (contains-many? opts :parent :corner :width :indices :similarity-fn)
        (SpTree. parent data corner width indices similarity-fn)
        (contains-many? opts :parent :data :corner :width :indices)
        (SpTree. parent data corner width indices)
        (contains-many? opts :indices :similarity-fn)
        (SpTree. data indices similarity-fn)
        (contains? opts :indices)
        (SpTree. data indices)
        :else
        (SpTree. data)))

(defn compute-edge-forces!
  "Compute edge forces using barns hut"
  [& {:keys [sp-tree row-p col-p val-p n pos-f]}]
  (doto sp-tree (.computeEdgeForces row-p col-p val-p n pos-f)))

(defn compute-non-edge-forces!
  "Compute non edge forces using barnes hut"
  [& {:keys [sp-tree point-idx theta negative-force sum-q]}]
  (doto sp-tree (.computeNonEdgeForces point-idx theta negative-force sum-q)))

(defn depth
  "The depth of the node"
  [sp-tree]
  (.depth sp-tree))

(defn get-boundary
  [sp-tree]
  (.getBoundary sp-tree))

(defn get-center-of-mass
  [sp-tree]
  (.getCenterOfMass sp-tree))

(defn get-children
  [sp-tree]
  (.getChildren sp-tree))

(defn get-cumulative-size
  [sp-tree]
  (.getCumSize sp-tree))

(defn get-d
  [sp-tree]
  (.getD sp-tree))

(defn get-idx
  [sp-tree]
  (.getIndex sp-tree))

(defn get-num-children
  [sp-tree]
  (.getNumChildren sp-tree))

(defn correct-structure?
  "Verifies the structure of the tree (does bounds checking on each node)"
  [sp-tree]
  (.isCorrect sp-tree))

(defn is-leaf?
  [sp-tree]
  (.isLeaf sp-tree))

(defn set-cummulative-size!
  [& {:keys [sp-tree size]}]
  (doto sp-tree (.setCumSize size)))

(defn set-num-children!
  [& {:keys [sp-tree n]}]
  (doto sp-tree (.setNumChildren n)))

(defn sub-divide!
  [sp-tree]
  (doto sp-tree (.subDivide)))
