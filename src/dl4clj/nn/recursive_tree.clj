(ns ^{:doc "Tree for a recursive neural tensor network based on Socher et al's work.
Implementation of the Tree class in dl4j.
NOTE this ns is a first attempt, I need to experiment/read more to accurately wrap the Tree class
 -- this needs to be refactored
see: https://deeplearning4j.org/doc/org/deeplearning4j/nn/layers/feedforward/autoencoder/recursive/Tree.html"}
    dl4clj.nn.recursive-tree
  (:import [org.deeplearning4j.nn.layers.feedforward.autoencoder.recursive Tree])
  (:require [dl4clj.utils :refer [contains-many?]]))

;; refactor
(defn new-tree
  "creates a new tree if supplied a list of tokens (strings), another tree, or
  a tree parent and a list of strings as the children (tokens)

  if only :tokens supplied, creates a tree with these tokens as the parents
  if only :tree supplied, clones the tree except for its children
  if both are supplied, :tree is the parent and :tokens are the children"
  [& {:keys [tokens tree]
      :as opts}]
  (cond (contains-many? opts :tokens :tree)
        (Tree. tree tokens)
        (contains? opts :tokens)
        (Tree. tokens)
        (contains? opts :tree)
        (Tree. tree)
        :else
        (assert false "you must supply a list of tokens and/or an existing tree structure")))
