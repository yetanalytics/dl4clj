(ns dl4clj.clustering.vptree.node
  (:import [org.deeplearning4j.clustering.vptree VPTree$Node]))

(defn new-vp-tree-node
  [& {:keys [idx threshold]}]
  (VPTree$Node. idx threshold))

(defn get-idx
  [node]
  (.getIndex node))

(defn get-left
  [vp-tree]
  (.getLeft vp-tree))

(defn get-right
  [vp-tree]
  (.getRight vp-tree))

(defn get-threshold
  [node]
  (.getThreshold node))

(defn set-idx!
  [& {:keys [node idx]}]
  (doto node (.setIndex idx)))

(defn set-left!
  [& {:keys [vp-tree left-node]}]
  (doto vp-tree (.setLeft left-node)))

(defn set-right!
  [& {:keys [vp-tree right-node]}]
  (doto vp-tree (.setRight right-node)))

(defn set-threshold!
  [& {:keys [node threshold]}]
  (doto node (.setThreshold threshold)))
