(ns dl4clj.nn.api.tree
  (:import [org.deeplearning4j.nn.layers.feedforward.autoencoder.recursive Tree])
  (:require [dl4clj.utils :refer [contains-many?]]))

(defn get-ancestor
  "returns the ancestor of the given tree.
  --note see if tree (this) and root are the same thing"
  [& {:keys [tree height root]}]
  (.ancestor tree height root))

(defn get-children
  "returns the children of a supplied tree"
  [tree]
  (.children tree))
;; there is also a .getChildren method
;; figure out how .children and .getChildren differ or if they do

(defn get-depth
  "returns the depth of the tree"
  [tree]
  (.depth tree))

(defn get-distance-between-nodes
  "returns the distance between node and subnode"
  [& {:keys [node subnode]}]
  (.depth node subnode))

(defn clone
  "clones the tree"
  [tree]
  (.clone tree))

(defn connect!
  "Connects the given trees and sets the parents of the children"
  [& {:keys [tree children]}]
  (doto tree
    (.connect children)))

(defn get-error
  "Returns the prediction error for this node"
  [node]
  (.error node))

(defn get-error-sum
  "Returns the total prediction error for this tree and its children"
  [tree]
  (.errorSum tree))

(defn first-child
  "returns the first child"
  [tree]
  (.firstChild tree))

(defn get-begin
  "test to determine doc string"
  [tree]
  (.getBegin tree))

(defn get-end
  "test to determine doc string"
  [tree]
  (.getEnd tree))

(defn get-head-word
  "test to determine doc string"
  [tree]
  (.getHeadWord tree))

(defn get-leaves
  "gets the leaves of the tree"
  [& {:keys [tree specific-leaves]
      :as opts}]
  (cond (contains-many? opts :tree :specific-leaves)
        (.getLeaves tree specific-leaves)
        (contains? opts :tree)
        (.getLeaves tree)
        :else
        (assert false "you must supply a tree to get its leaves")))

(defn get-tokens
  "returns a list of tokens associated with this tree"
  [tree]
  (.getTokens tree))

(defn get-node-type
  [node]
  (.getType node))

(defn gold-label
  "test to determine doc string"
  [tree]
  (.goldLabel tree))

(defn get-tags
  [word]
  (.tags word))

(defn to-string
  [tree]
  (.toString tree))

(defn get-value
  [node]
  (.value node))

(defn as-vector
  [tree]
  (.vector tree))

(defn yield
  "Returns all of the labels for this node and all of its children (recursively)"
  [node]
  (.yield node))

(defn is-leaf?
  "Returns whether the node has any children or not"
  [node]
  (.isLeaf node))

(defn is-pre-terminal?
  "tests if Node has one child that is a leaf"
  [node]
  (.isPreTerminal node))

(defn label
  "test to determine doc string"
  [tree]
  (.label tree))

(defn get-last-child
  "returns the last child of the tree"
  [tree]
  (.lastChild tree))

(defn get-parent
  "returns the parent of the given node"
  [& {:keys [node root]
      :as opts}]
  (cond (contains-many? opts :node :root)
        (.parent node root)
        (contains? opts :node)
        (.parent node)
        :else
        (assert false "you must supply a node to get the parent")))

(defn prediction
  "test to see if tree is the correct arg here
  predicts a label for a given something..."
  [tree]
  (.prediction tree))

(defn set-begin!
  [& {:keys [tree begin-idx]}]
  (doto tree
    (.setBegin begin-idx)))

(defn set-end!
  [& {:keys [tree end-idx]}]
  (doto tree
    (.setEnd tree end-idx)))

(defn set-error!
  [& {:keys [tree error]}]
  (doto tree
    (.setError error)))

(defn set-gold-label!
  [& {:keys [tree gold-label]}]
  (doto tree
    (.setGoldLabel tree gold-label)))

(defn set-head-word!
  [& {:keys [tree head-word]}]
  (doto tree
    (.setHeadWord head-word)))

(defn set-label!
  [& {:keys [tree label]}]
  (doto tree
    (.setLabel label)))

(defn set-parrent!
  [& {:keys [tree parent]}]
  (doto tree
    (.setParent parent)))

(defn set-parse!
  [& {:keys [tree parse]}]
  (doto tree
    (.setParse parse)))

(defn set-prediction!
  [& {:keys [tree prediction]}]
  (doto tree
    (.setPrediction prediction)))

(defn set-tags!
  [& {:keys [tree tags]}]
  (doto tree
    (.setTags tags)))

(defn set-tokens!
  [& {:keys [tree tokens]}]
  (doto tree
    (.setTokens tokens)))

(defn set-type!
  [& {:keys [node node-type]}]
  (doto node
    (.setType node-type)))

(defn set-value!
  [& {:keys [node value]}]
  (doto node
    (.setValue value)))

(defn set-vector!
  [& {:keys [node v]}]
  (doto node
    (.setVector v)))
