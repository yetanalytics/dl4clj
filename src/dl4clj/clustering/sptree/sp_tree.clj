(ns dl4clj.clustering.sptree.sp-tree
  (:import [org.deeplearning4j.clustering.sptree SpTree])
  (:require [dl4clj.utils :refer [contains-many?]]))

(defn new-sptree
  [& {:keys [data parent indices similarity-fn corner width]
      :as opts}]
  (assert (contains? opts :data) "you must provide data to create the tree with")
  (cond (contains-many? opts :parent :corner :width :indices :similarity-fn)
        (SpTree. parent data corner width indices similarity-fn)



        )

  )
