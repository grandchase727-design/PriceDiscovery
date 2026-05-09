"""
GraphRAG Engine for Price Discovery Scanner
─────────────────────────────────────────────
Builds a knowledge graph from scanner results, detects communities,
and generates structured insights for multi-hop reasoning.

Node types: Ticker, Category, Theme, Classification, Benchmark, AssetType
Edge types: BELONGS_TO, HAS_THEME, CLASSIFIED_AS, BENCHMARKED_BY,
            SCORE_SIMILAR, THEME_PEER, RETURN_CORRELATED, TRANSITIONED
"""

import numpy as np
import networkx as nx
from collections import defaultdict, Counter
from itertools import combinations

try:
    import community.community_louvain as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False


###############################################################################
# GRAPH BUILDER
###############################################################################

class PriceDiscoveryGraph:
    """Knowledge graph built from scanner results."""

    # ── Node type prefixes (namespace isolation) ──
    P_TICK = "T:"    # Ticker
    P_CAT  = "C:"    # Category
    P_THM  = "TH:"   # Theme
    P_CLS  = "CL:"   # Classification
    P_BENCH = "B:"   # Benchmark
    P_ASSET = "A:"   # AssetType (ETF / Stock)
    P_BUCKET = "BK:"  # ScoreBucket

    SCORE_SIMILAR_THRESHOLD = 5.0   # composite ±5
    RETURN_CORR_THRESHOLD = 0.65    # ret correlation > 0.65

    def __init__(self):
        self.G = nx.Graph()
        self.ticker_data = {}       # ticker -> result dict
        self.communities = {}       # ticker -> community_id
        self.community_stats = {}   # community_id -> stats dict
        self.insights = []          # list of insight strings
        self.etf_tickers = set()
        self.stock_tickers = set()

    # ─────────────────────────────────────────────────────────────────
    # BUILD
    # ─────────────────────────────────────────────────────────────────

    def build(self, results, stock_themes, etf_universe, stock_universe,
              category_benchmark, stock_benchmark, history_7d=None):
        """Build full knowledge graph from scanner results."""

        # Identify ETF vs Stock tickers
        for cat, data in etf_universe.items():
            self.etf_tickers.update(data['tickers'].keys())
        for cat, data in stock_universe.items():
            self.stock_tickers.update(data['tickers'].keys())

        # Index results
        for r in results:
            self.ticker_data[r['ticker']] = r

        self._add_structural_nodes(results, stock_themes,
                                    category_benchmark, stock_benchmark)
        self._add_ticker_similarity_edges(results)
        self._add_theme_peer_edges(stock_themes)
        self._add_return_correlation_edges(results)
        if history_7d:
            self._add_transition_edges(history_7d)

        print(f"   📊 Graph: {self.G.number_of_nodes()} nodes, "
              f"{self.G.number_of_edges()} edges")
        return self

    def _add_structural_nodes(self, results, stock_themes,
                               category_benchmark, stock_benchmark):
        """Add all entity nodes and structural edges."""
        G = self.G

        # AssetType nodes
        G.add_node(self.P_ASSET + "ETF", type="AssetType", label="ETF")
        G.add_node(self.P_ASSET + "Stock", type="AssetType", label="Stock")

        # ScoreBucket nodes
        for bk in ["0-30", "30-50", "50-70", "70-100"]:
            G.add_node(self.P_BUCKET + bk, type="ScoreBucket", label=bk)

        all_benchmarks = {**category_benchmark, **stock_benchmark}

        for r in results:
            tk = r['ticker']
            tk_id = self.P_TICK + tk
            cat_id = self.P_CAT + r['category']
            cls_id = self.P_CLS + r['classification']
            asset = "Stock" if tk in self.stock_tickers else "ETF"
            asset_id = self.P_ASSET + asset

            # Determine bucket
            comp = r['composite']
            if comp < 30: bk = "0-30"
            elif comp < 50: bk = "30-50"
            elif comp < 70: bk = "50-70"
            else: bk = "70-100"
            bk_id = self.P_BUCKET + bk

            # Ticker node
            G.add_node(tk_id, type="Ticker", label=tk, asset=asset,
                       composite=comp, classification=r['classification'],
                       tcs=r['tcs'], tfs=r['tfs'], oer=r['oer'], rss=r['rss'],
                       tcs_s=r.get('tcs_short', 0), tcs_l=r.get('tcs_long', 0),
                       tfs_s=r.get('tfs_short', 0), tfs_l=r.get('tfs_long', 0),
                       eligible=r['eligible'], trend_age=r['trend_age'],
                       ret_1w=r.get('ret_1w', 0), ret_1m=r.get('ret_1m', 0),
                       ret_3m=r.get('ret_3m', 0))

            # Category node
            if not G.has_node(cat_id):
                bench_tk = all_benchmarks.get(r['category'], 'SPY')
                G.add_node(cat_id, type="Category", label=r['category'])
                bench_id = self.P_BENCH + bench_tk
                if not G.has_node(bench_id):
                    G.add_node(bench_id, type="Benchmark", label=bench_tk)
                G.add_edge(cat_id, bench_id, relation="BENCHMARKED_BY")

            # Classification node
            if not G.has_node(cls_id):
                G.add_node(cls_id, type="Classification", label=r['classification'])

            # Structural edges
            G.add_edge(tk_id, cat_id, relation="BELONGS_TO")
            G.add_edge(tk_id, cls_id, relation="CLASSIFIED_AS")
            G.add_edge(tk_id, asset_id, relation="IS_TYPE")
            G.add_edge(tk_id, bk_id, relation="IN_BUCKET")

            # Theme edge (stocks only)
            theme = stock_themes.get(tk)
            if theme:
                thm_id = self.P_THM + theme
                if not G.has_node(thm_id):
                    G.add_node(thm_id, type="Theme", label=theme)
                    # Theme belongs to category
                    G.add_edge(thm_id, cat_id, relation="CHILD_OF")
                G.add_edge(tk_id, thm_id, relation="HAS_THEME")

    def _add_ticker_similarity_edges(self, results):
        """Add edges between tickers with similar composite scores."""
        tickers = [(r['ticker'], r['composite']) for r in results]
        # Sort by composite for efficient neighbor search
        tickers.sort(key=lambda x: x[1])
        threshold = self.SCORE_SIMILAR_THRESHOLD
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                if tickers[j][1] - tickers[i][1] > threshold:
                    break
                t1, t2 = tickers[i][0], tickers[j][0]
                self.G.add_edge(self.P_TICK + t1, self.P_TICK + t2,
                                relation="SCORE_SIMILAR",
                                weight=1.0 - abs(tickers[i][1] - tickers[j][1]) / threshold)

    def _add_theme_peer_edges(self, stock_themes):
        """Add edges between tickers sharing the same theme."""
        theme_groups = defaultdict(list)
        for tk, theme in stock_themes.items():
            if self.P_TICK + tk in self.G:
                theme_groups[theme].append(tk)

        for theme, tickers in theme_groups.items():
            if len(tickers) < 2:
                continue
            for t1, t2 in combinations(tickers, 2):
                if not self.G.has_edge(self.P_TICK + t1, self.P_TICK + t2):
                    self.G.add_edge(self.P_TICK + t1, self.P_TICK + t2,
                                    relation="THEME_PEER", theme=theme, weight=0.8)

    def _add_return_correlation_edges(self, results):
        """Add edges between tickers with correlated returns."""
        # Use 1W/1M/3M returns as a 3-dimensional vector per ticker
        vectors = {}
        for r in results:
            ret_vec = [r.get('ret_1w', 0), r.get('ret_1m', 0), r.get('ret_3m', 0)]
            if any(v != 0 for v in ret_vec):
                vectors[r['ticker']] = np.array(ret_vec, dtype=float)

        tickers = list(vectors.keys())
        if len(tickers) < 2:
            return

        # Normalize vectors
        mat = np.array([vectors[t] for t in tickers])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat_norm = mat / norms

        # Compute cosine similarity matrix (only upper triangle)
        threshold = self.RETURN_CORR_THRESHOLD
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                sim = float(np.dot(mat_norm[i], mat_norm[j]))
                if sim > threshold:
                    t1_id = self.P_TICK + tickers[i]
                    t2_id = self.P_TICK + tickers[j]
                    if not self.G.has_edge(t1_id, t2_id):
                        self.G.add_edge(t1_id, t2_id,
                                        relation="RETURN_CORRELATED",
                                        weight=sim)

    def _add_transition_edges(self, history_7d):
        """Add classification transition edges from 7-day history."""
        for ticker, hist in history_7d.items():
            if len(hist) < 2:
                continue
            tk_id = self.P_TICK + ticker
            if tk_id not in self.G:
                continue
            for i in range(len(hist) - 1):
                cls_from = hist[i].get('class', '')
                cls_to = hist[i + 1].get('class', '')
                if cls_from and cls_to and cls_from != cls_to:
                    from_id = self.P_CLS + cls_from
                    to_id = self.P_CLS + cls_to
                    if from_id in self.G and to_id in self.G:
                        key = (from_id, to_id)
                        if self.G.has_edge(*key):
                            self.G[from_id][to_id]['weight'] = \
                                self.G[from_id][to_id].get('weight', 0) + 1
                        else:
                            self.G.add_edge(from_id, to_id,
                                            relation="TRANSITIONS_TO", weight=1)

    # ─────────────────────────────────────────────────────────────────
    # COMMUNITY DETECTION
    # ─────────────────────────────────────────────────────────────────

    def detect_communities(self):
        """Run Louvain community detection on ticker subgraph."""
        if not HAS_LOUVAIN:
            print("   ⚠️ python-louvain not installed, skipping community detection")
            return self

        # Extract ticker-only subgraph with weighted edges
        ticker_nodes = [n for n, d in self.G.nodes(data=True) if d.get('type') == 'Ticker']
        sub = self.G.subgraph(ticker_nodes).copy()

        if len(sub) < 3:
            return self

        partition = community_louvain.best_partition(sub, random_state=42)
        self.communities = {n.replace(self.P_TICK, ''): cid
                            for n, cid in partition.items()}

        # Aggregate community stats
        comm_groups = defaultdict(list)
        for tk, cid in self.communities.items():
            if tk in self.ticker_data:
                comm_groups[cid].append(self.ticker_data[tk])

        self.community_stats = {}
        for cid, members in sorted(comm_groups.items()):
            if not members:
                continue
            cats = Counter(m['category'] for m in members)
            clss = Counter(m['classification'] for m in members)
            assets = Counter('Stock' if m['ticker'] in self.stock_tickers else 'ETF'
                             for m in members)
            themes = Counter()
            for m in members:
                node = self.G.nodes.get(self.P_TICK + m['ticker'], {})
                # Find theme via graph edges
                for neighbor in self.G.neighbors(self.P_TICK + m['ticker']):
                    nd = self.G.nodes.get(neighbor, {})
                    if nd.get('type') == 'Theme':
                        themes[nd['label']] += 1

            avg_comp = np.mean([m['composite'] for m in members])
            avg_ret1m = np.mean([m.get('ret_1m', 0) for m in members])
            avg_ret3m = np.mean([m.get('ret_3m', 0) for m in members])
            n_eligible = sum(1 for m in members if m['eligible'])

            self.community_stats[cid] = {
                'n': len(members),
                'tickers': sorted([m['ticker'] for m in members]),
                'top_categories': cats.most_common(5),
                'classification_dist': dict(clss),
                'asset_mix': dict(assets),
                'top_themes': themes.most_common(5),
                'avg_composite': round(avg_comp, 1),
                'avg_ret_1m': round(avg_ret1m, 2),
                'avg_ret_3m': round(avg_ret3m, 2),
                'n_eligible': n_eligible,
                'eligible_pct': round(n_eligible / len(members) * 100, 1),
                'dominant_class': clss.most_common(1)[0][0] if clss else 'N/A',
            }

        n_comm = len(self.community_stats)
        print(f"   🔗 Communities: {n_comm} detected across "
              f"{len(self.communities)} tickers")
        return self

    # ─────────────────────────────────────────────────────────────────
    # ANALYSIS MODULES
    # ─────────────────────────────────────────────────────────────────

    def analyze_all(self):
        """Run all analysis modules and collect insights."""
        self.insights = []
        self.viz_data = {}  # structured data for dashboard charts
        self._analyze_community_character()
        self._analyze_theme_propagation()
        self._analyze_etf_stock_divergence()
        self._analyze_leader_lagger()
        self._analyze_category_entropy()
        self._analyze_cross_category_flow()
        return self

    def _analyze_community_character(self):
        """Characterize each community: risk-on/off, sector tilt, momentum."""
        bullish_cls = {'🟢 CONTINUATION', '🔵 FORMATION', '🔵 RECOVERY'}
        bearish_cls = {'⬇️ DOWNTREND', '🟤 FADING', '🟣 COUNTER_RALLY'}
        comm_viz = []  # for dashboard chart

        for cid, stats in self.community_stats.items():
            if stats['n'] < 3:
                continue
            dom_cls = stats['dominant_class']
            top_cats = [c[0] for c in stats['top_categories'][:3]]
            top_themes = [t[0] for t in stats['top_themes'][:3]]

            bull_pct = sum(stats['classification_dist'].get(c, 0) for c in bullish_cls) / stats['n'] * 100
            bear_pct = sum(stats['classification_dist'].get(c, 0) for c in bearish_cls) / stats['n'] * 100

            if bull_pct > 60:
                character = "Risk-On"
            elif bear_pct > 60:
                character = "Risk-Off"
            elif stats['avg_composite'] > 60:
                character = "Strong Momentum"
            elif stats['avg_composite'] < 40:
                character = "Weak/Distressed"
            else:
                character = "Mixed/Transitional"

            theme_str = ", ".join(top_themes) if top_themes else "N/A"
            cat_str = ", ".join(c.replace("STK_", "") for c in top_cats)

            comm_viz.append({
                'community': cid, 'character': character, 'n': stats['n'],
                'bull_pct': round(bull_pct, 1), 'bear_pct': round(bear_pct, 1),
                'avg_composite': stats['avg_composite'],
                'avg_ret_1m': stats['avg_ret_1m'], 'avg_ret_3m': stats['avg_ret_3m'],
                'eligible_pct': stats['eligible_pct'],
                'top_categories': cat_str, 'top_themes': theme_str,
                'dominant_class': dom_cls,
            })

            self.insights.append({
                'type': 'community_character',
                'community': cid,
                'character': character,
                'detail': (f"Community {cid} ({stats['n']} tickers): {character} | "
                           f"Avg Composite={stats['avg_composite']} | "
                           f"Dominant={dom_cls} | "
                           f"Bull/Bear={bull_pct:.0f}%/{bear_pct:.0f}% | "
                           f"Categories=[{cat_str}] | Themes=[{theme_str}] | "
                           f"1M Ret={stats['avg_ret_1m']:.1f}% | "
                           f"3M Ret={stats['avg_ret_3m']:.1f}%"),
            })

        self.viz_data['community_character'] = comm_viz

    def _analyze_theme_propagation(self):
        """Detect momentum propagation within themes using 1W/1M/3M snapshots."""
        theme_tickers = defaultdict(list)
        for tk, data in self.ticker_data.items():
            for neighbor in self.G.neighbors(self.P_TICK + tk):
                nd = self.G.nodes.get(neighbor, {})
                if nd.get('type') == 'Theme':
                    theme_tickers[nd['label']].append(data)

        prop_viz = []  # for dashboard funnel/sankey chart

        for theme, members in theme_tickers.items():
            if len(members) < 3:
                continue

            cls_counts = Counter(m['classification'] for m in members)
            has_cont = cls_counts.get('🟢 CONTINUATION', 0)
            has_form = cls_counts.get('🔵 FORMATION', 0) + cls_counts.get('🔵 RECOVERY', 0)
            has_neutral = cls_counts.get('🟠 NEUTRAL', 0) + cls_counts.get('🟡 CONSOLIDATION', 0)
            has_bearish = sum(cls_counts.get(c, 0) for c in
                             ('⬇️ DOWNTREND', '🟤 FADING', '🟣 COUNTER_RALLY', '⚠️ WEAKENING'))

            leaders = [m for m in members if m['classification'] == '🟢 CONTINUATION']
            forming = [m for m in members
                       if m['classification'] in ('🔵 FORMATION', '🔵 RECOVERY')]
            neutral = [m for m in members
                       if m['classification'] in ('🟠 NEUTRAL', '🟡 CONSOLIDATION')]
            pullback = [m for m in members if m['classification'] == '🔶 PULLBACK']
            bearish = [m for m in members
                       if m['classification'] in ('⬇️ DOWNTREND', '🟤 FADING',
                                                   '🟣 COUNTER_RALLY', '⚠️ WEAKENING')]

            # Build per-ticker row for viz
            for m in members:
                prop_viz.append({
                    'theme': theme, 'ticker': m['ticker'],
                    'composite': m['composite'],
                    'classification': m['classification'],
                    'stage': ('Leader' if m in leaders else
                              'Forming' if m in forming else
                              'Neutral' if m in neutral else
                              'Pullback' if m in pullback else 'Bearish'),
                    'ret_1m': m.get('ret_1m', 0),
                })

            if has_cont > 0 and has_form > 0:
                self.insights.append({
                    'type': 'theme_propagation',
                    'theme': theme,
                    'stage': 'active_propagation',
                    'detail': (f"Theme [{theme}]: Momentum propagation detected | "
                               f"Leaders(CONT)={[m['ticker'] for m in leaders[:5]]} | "
                               f"Forming={[m['ticker'] for m in forming[:5]]} | "
                               f"Lagging(potential)={[m['ticker'] for m in neutral[:5]]}"),
                })
            elif has_form >= 2 and has_cont == 0:
                self.insights.append({
                    'type': 'theme_propagation',
                    'theme': theme,
                    'stage': 'early_formation',
                    'detail': (f"Theme [{theme}]: Early formation (no leaders yet) | "
                               f"Forming={[m['ticker'] for m in forming[:5]]}"),
                })

        self.viz_data['theme_propagation'] = prop_viz

    def _analyze_etf_stock_divergence(self):
        """Find cases where ETF classification diverges from constituent stocks."""
        cat_mapping = {
            'EQ_Technology': {
                'XLK': 'STK_Technology', 'SMH': 'STK_Technology',
            },
            'EQ_Healthcare': {
                'XLV': 'STK_Healthcare',
            },
            'EQ_Financials': {
                'XLF': 'STK_Financials',
            },
            'EQ_ConsDisc': {
                'XLY': 'STK_ConsDisc',
            },
            'EQ_ConsStaples': {
                'XLP': 'STK_ConsStaples',
            },
            'EQ_Industrials': {
                'XLI': 'STK_Industrials',
            },
            'EQ_Energy': {
                'XLE': 'STK_Energy',
            },
            'EQ_Materials': {
                'XLB': 'STK_Materials',
            },
            'EQ_Utilities': {
                'XLU': 'STK_Utilities',
            },
            'EQ_RealEstate': {
                'XLRE': 'STK_RealEstate',
            },
            'EQ_CommServices': {
                'XLC': 'STK_CommServices',
            },
        }

        bullish = {'🟢 CONTINUATION', '🔵 FORMATION', '🔵 RECOVERY', '🟡 OVEREXTENDED'}
        bearish = {'⬇️ DOWNTREND', '🟤 FADING', '🟣 COUNTER_RALLY', '⚠️ WEAKENING'}
        div_viz = []  # for dashboard comparison chart

        for etf_cat, bench_map in cat_mapping.items():
            for etf_tk, stk_cat in bench_map.items():
                etf_data = self.ticker_data.get(etf_tk)
                if not etf_data:
                    continue

                stk_members = [d for tk, d in self.ticker_data.items()
                               if d['category'] == stk_cat]
                if not stk_members:
                    continue

                etf_cls = etf_data['classification']
                stk_cls_counts = Counter(m['classification'] for m in stk_members)

                stk_bull_pct = sum(stk_cls_counts.get(c, 0) for c in bullish) / len(stk_members) * 100
                stk_bear_pct = sum(stk_cls_counts.get(c, 0) for c in bearish) / len(stk_members) * 100
                stk_avg_comp = np.mean([m['composite'] for m in stk_members])

                div_viz.append({
                    'etf': etf_tk, 'stock_category': stk_cat,
                    'etf_class': etf_cls, 'etf_composite': etf_data['composite'],
                    'stock_avg_composite': round(stk_avg_comp, 1),
                    'stock_bull_pct': round(stk_bull_pct, 1),
                    'stock_bear_pct': round(stk_bear_pct, 1),
                    'stock_n': len(stk_members),
                    'stock_cls_dist': dict(stk_cls_counts),
                    'divergent': (etf_cls in bullish and stk_bear_pct > 40) or
                                 (etf_cls in bearish and stk_bull_pct > 40),
                })

                etf_bull = etf_cls in bullish
                if etf_bull and stk_bear_pct > 40:
                    self.insights.append({
                        'type': 'etf_stock_divergence', 'etf': etf_tk,
                        'stock_category': stk_cat,
                        'detail': (f"ETF-Stock Divergence: {etf_tk}={etf_cls} but "
                                   f"{stk_bear_pct:.0f}% of {stk_cat} stocks are bearish | "
                                   f"ETF signal may be driven by few large-caps"),
                    })
                etf_bear = etf_cls in bearish
                if etf_bear and stk_bull_pct > 40:
                    self.insights.append({
                        'type': 'etf_stock_divergence', 'etf': etf_tk,
                        'stock_category': stk_cat,
                        'detail': (f"ETF-Stock Divergence: {etf_tk}={etf_cls} but "
                                   f"{stk_bull_pct:.0f}% of {stk_cat} stocks are bullish | "
                                   f"Hidden strength in individual stocks"),
                    })

        self.viz_data['etf_stock_divergence'] = div_viz

    def _analyze_leader_lagger(self):
        """Within each theme, find leaders and potential catch-up candidates."""
        theme_tickers = defaultdict(list)
        for tk, data in self.ticker_data.items():
            for neighbor in self.G.neighbors(self.P_TICK + tk):
                nd = self.G.nodes.get(neighbor, {})
                if nd.get('type') == 'Theme':
                    theme_tickers[nd['label']].append(data)

        for theme, members in theme_tickers.items():
            if len(members) < 3:
                continue

            sorted_members = sorted(members, key=lambda x: -x['composite'])
            leader = sorted_members[0]
            laggers = [m for m in sorted_members
                       if m['eligible'] is False
                       and m['composite'] > 40
                       and m['classification'] not in ('⬇️ DOWNTREND', '🟤 FADING')]

            if leader['eligible'] and leader['composite'] > 65 and laggers:
                self.insights.append({
                    'type': 'leader_lagger',
                    'theme': theme,
                    'detail': (f"Theme [{theme}] Leader-Lagger Gap: "
                               f"Leader={leader['ticker']}(Comp={leader['composite']:.0f}, "
                               f"{leader['classification']}) | "
                               f"Catch-up candidates={[l['ticker'] for l in laggers[:3]]} "
                               f"(near-eligible, potential upgrades)"),
                })

    def _analyze_category_entropy(self):
        """Measure classification dispersion within each category."""
        cat_groups = defaultdict(list)
        for data in self.ticker_data.values():
            cat_groups[data['category']].append(data['classification'])

        entropy_viz = []  # for dashboard bar chart

        for cat, cls_list in cat_groups.items():
            if len(cls_list) < 5:
                continue
            counts = Counter(cls_list)
            probs = np.array(list(counts.values()), dtype=float)
            probs /= probs.sum()
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 0

            label = ("HIGH — Opinion Split" if norm_entropy > 0.85
                     else "LOW — Strong Consensus" if norm_entropy < 0.40
                     else "MODERATE")

            entropy_viz.append({
                'category': cat.replace("STK_", ""),
                'entropy': round(norm_entropy, 3),
                'n': len(cls_list),
                'label': label,
                'dominant': counts.most_common(1)[0][0] if counts else 'N/A',
                'dominant_pct': round(counts.most_common(1)[0][1] / len(cls_list) * 100, 1) if counts else 0,
                'cls_dist': dict(counts),
            })

            if norm_entropy > 0.85:
                self.insights.append({
                    'type': 'category_entropy', 'category': cat,
                    'entropy': round(norm_entropy, 3),
                    'detail': (f"Category [{cat}]: HIGH classification entropy "
                               f"({norm_entropy:.2f}) — market opinion is split | "
                               f"Distribution: {dict(counts)} | "
                               f"Selective stock-picking required"),
                })
            elif norm_entropy < 0.40 and len(cls_list) >= 10:
                dom = counts.most_common(1)[0]
                self.insights.append({
                    'type': 'category_entropy', 'category': cat,
                    'entropy': round(norm_entropy, 3),
                    'detail': (f"Category [{cat}]: LOW entropy ({norm_entropy:.2f}) — "
                               f"strong consensus | "
                               f"Dominant: {dom[0]} ({dom[1]}/{len(cls_list)}) | "
                               f"Category-level bet viable"),
                })

        self.viz_data['category_entropy'] = sorted(entropy_viz, key=lambda x: -x['entropy'])

    def _analyze_cross_category_flow(self):
        """Detect momentum flow across categories using community bridges."""
        if not self.communities:
            return

        # Find communities that span multiple categories
        for cid, stats in self.community_stats.items():
            cats = stats['top_categories']
            if len(cats) < 2:
                continue

            # Check if bridge community with mixed ETF/Stock
            asset_mix = stats.get('asset_mix', {})
            has_etf = asset_mix.get('ETF', 0) > 0
            has_stock = asset_mix.get('Stock', 0) > 0

            if has_etf and has_stock and stats['n'] >= 5:
                cat_names = [c[0].replace("STK_", "") for c in cats[:4]]
                self.insights.append({
                    'type': 'cross_category_flow',
                    'community': cid,
                    'detail': (f"Cross-Category Community {cid}: "
                               f"Bridges [{', '.join(cat_names)}] | "
                               f"ETF/Stock mix={asset_mix} | "
                               f"{stats['dominant_class']} dominant | "
                               f"Avg Composite={stats['avg_composite']} | "
                               f"Suggests correlated momentum across asset types"),
                })

    # ─────────────────────────────────────────────────────────────────
    # MULTI-HOP QUERIES
    # ─────────────────────────────────────────────────────────────────

    def query_impact_radius(self, ticker, hops=2):
        """Find all tickers within N hops that would be affected."""
        tk_id = self.P_TICK + ticker
        if tk_id not in self.G:
            return {'error': f'{ticker} not in graph'}

        affected = {}
        visited = {tk_id}
        frontier = [(tk_id, 0)]

        while frontier:
            node, depth = frontier.pop(0)
            if depth >= hops:
                continue
            for neighbor in self.G.neighbors(node):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                nd = self.G.nodes[neighbor]
                edge_data = self.G[node][neighbor]
                rel = edge_data.get('relation', '')

                if nd.get('type') == 'Ticker':
                    tk_name = neighbor.replace(self.P_TICK, '')
                    td = self.ticker_data.get(tk_name, {})
                    affected[tk_name] = {
                        'hop': depth + 1,
                        'relation': rel,
                        'classification': td.get('classification', 'N/A'),
                        'composite': td.get('composite', 0),
                        'category': td.get('category', 'N/A'),
                    }
                    frontier.append((neighbor, depth + 1))
                elif nd.get('type') in ('Theme', 'Category'):
                    frontier.append((neighbor, depth + 1))

        return {
            'source': ticker,
            'source_data': self.ticker_data.get(ticker, {}),
            'affected': dict(sorted(affected.items(),
                                     key=lambda x: (x[1]['hop'], -x[1]['composite']))),
            'total_affected': len(affected),
        }

    def query_theme_status(self, theme):
        """Get full status of a theme across all categories."""
        thm_id = self.P_THM + theme
        if thm_id not in self.G:
            return {'error': f'Theme "{theme}" not in graph'}

        members = []
        for neighbor in self.G.neighbors(thm_id):
            nd = self.G.nodes[neighbor]
            if nd.get('type') == 'Ticker':
                tk = neighbor.replace(self.P_TICK, '')
                if tk in self.ticker_data:
                    members.append(self.ticker_data[tk])

        if not members:
            return {'theme': theme, 'members': []}

        cls_dist = Counter(m['classification'] for m in members)
        cats = Counter(m['category'] for m in members)
        avg_comp = np.mean([m['composite'] for m in members])

        return {
            'theme': theme,
            'n': len(members),
            'avg_composite': round(avg_comp, 1),
            'classification_dist': dict(cls_dist),
            'categories': dict(cats),
            'members': sorted([{
                'ticker': m['ticker'], 'composite': m['composite'],
                'classification': m['classification'], 'eligible': m['eligible'],
            } for m in members], key=lambda x: -x['composite']),
        }

    def query_formation_pipeline(self):
        """Find all themes with active FORMATION/RECOVERY tickers and their pipeline."""
        pipeline = {}
        theme_tickers = defaultdict(list)

        for tk, data in self.ticker_data.items():
            for neighbor in self.G.neighbors(self.P_TICK + tk):
                nd = self.G.nodes.get(neighbor, {})
                if nd.get('type') == 'Theme':
                    theme_tickers[nd['label']].append(data)

        for theme, members in theme_tickers.items():
            forming = [m for m in members
                       if m['classification'] in ('🔵 FORMATION', '🔵 RECOVERY')]
            if not forming:
                continue

            continuing = [m for m in members if m['classification'] == '🟢 CONTINUATION']
            neutral = [m for m in members
                       if m['classification'] in ('🟠 NEUTRAL', '🟡 CONSOLIDATION')]
            pullback = [m for m in members if m['classification'] == '🔶 PULLBACK']

            pipeline[theme] = {
                'forming': [m['ticker'] for m in forming],
                'continuing': [m['ticker'] for m in continuing],
                'neutral_queue': [m['ticker'] for m in neutral],
                'pullback_candidates': [m['ticker'] for m in pullback],
                'total': len(members),
                'momentum_breadth': round((len(forming) + len(continuing)) / len(members) * 100, 1),
            }

        return dict(sorted(pipeline.items(), key=lambda x: -x[1]['momentum_breadth']))

    # ─────────────────────────────────────────────────────────────────
    # EXPORT
    # ─────────────────────────────────────────────────────────────────

    def export_for_llm(self):
        """Export graph insights as structured text for LLM prompt injection."""
        lines = []
        lines.append("=" * 80)
        lines.append("GRAPHRAG ANALYSIS: Knowledge Graph Insights")
        lines.append("=" * 80)

        # Community summaries
        lines.append("\n## COMMUNITY STRUCTURE")
        lines.append(f"Total communities detected: {len(self.community_stats)}")
        for cid, stats in sorted(self.community_stats.items(),
                                  key=lambda x: -x[1]['n']):
            if stats['n'] < 3:
                continue
            lines.append(f"\n### Community {cid} ({stats['n']} tickers)")
            lines.append(f"  Character: {stats['dominant_class']}")
            lines.append(f"  Avg Composite: {stats['avg_composite']}")
            lines.append(f"  Eligible: {stats['n_eligible']}/{stats['n']} ({stats['eligible_pct']}%)")
            lines.append(f"  1M Return: {stats['avg_ret_1m']:.1f}% | 3M Return: {stats['avg_ret_3m']:.1f}%")
            top_cats = ", ".join(f"{c[0]}({c[1]})" for c in stats['top_categories'][:3])
            lines.append(f"  Top Categories: {top_cats}")
            if stats['top_themes']:
                top_thm = ", ".join(f"{t[0]}({t[1]})" for t in stats['top_themes'][:5])
                lines.append(f"  Top Themes: {top_thm}")
            lines.append(f"  Class Dist: {stats['classification_dist']}")
            lines.append(f"  Tickers: {stats['tickers'][:15]}{'...' if stats['n'] > 15 else ''}")

        # Key insights
        lines.append("\n" + "=" * 80)
        lines.append("## KEY INSIGHTS")
        for insight in self.insights:
            lines.append(f"\n[{insight['type'].upper()}] {insight['detail']}")

        # Formation pipeline
        pipeline = self.query_formation_pipeline()
        if pipeline:
            lines.append("\n" + "=" * 80)
            lines.append("## FORMATION PIPELINE (Themes with Active Breakouts)")
            for theme, data in list(pipeline.items())[:15]:
                lines.append(f"\n  {theme} (breadth={data['momentum_breadth']}%)")
                lines.append(f"    Forming: {data['forming']}")
                if data['continuing']:
                    lines.append(f"    Continuing: {data['continuing']}")
                if data['neutral_queue']:
                    lines.append(f"    Queue (potential): {data['neutral_queue'][:5]}")

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)

    def get_summary_stats(self):
        """Return summary statistics for dashboard display."""
        n_nodes = self.G.number_of_nodes()
        n_edges = self.G.number_of_edges()
        type_counts = Counter(d.get('type', 'Unknown')
                              for _, d in self.G.nodes(data=True))
        rel_counts = Counter(d.get('relation', 'Unknown')
                             for _, _, d in self.G.edges(data=True))
        return {
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'node_types': dict(type_counts),
            'edge_types': dict(rel_counts),
            'n_communities': len(self.community_stats),
            'n_insights': len(self.insights),
        }
