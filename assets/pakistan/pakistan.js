/* Source: Pakistan charts. Reads the page's embedded data (#pv-data) and draws
   five small d3 charts. Every number drawn comes from that data. */
(function () {
  'use strict';
  var holder = document.getElementById('pv-data');
  if (!holder || typeof d3 === 'undefined') return;
  var PV;
  try { PV = JSON.parse(holder.textContent); } catch (e) { return; }

  var C1 = 'var(--pv-accent)', C2 = 'var(--pv-accent-2)', CM = 'var(--pv-chart-muted)';
  var GRID = 'var(--pv-grid)', AXIS = 'var(--pv-line)', REF = '#8a9299';

  function num(v) { return (v === null || v === undefined || v === '') ? null : Number(v); }
  function indicator(key) {
    return (PV.indicators || []).find(function (r) { return r.key === key; }) || null;
  }
  function textWidth(svg, s) {
    var t = svg.append('text').text(s);
    var w = t.node().getComputedTextLength();
    t.remove();
    return w;
  }
  function cleanAxis(g) {
    g.attr('font-size', null).attr('font-family', null);
    g.select('.domain').remove();
    g.selectAll('line').remove();
  }
  function legend(el, items) {
    var lg = document.createElement('div');
    lg.className = 'pv-legend';
    items.forEach(function (it) {
      var s = document.createElement('span');
      var sw = document.createElement('i');
      sw.className = 'pv-sw' + (it.goal ? ' pv-sw-goal' : '');
      sw.style.background = it.color;
      s.appendChild(sw);
      s.appendChild(document.createTextNode(it.name));
      lg.appendChild(s);
    });
    el.appendChild(lg);
  }
  function makeSvg(el, H, label) {
    var W = Math.max(280, el.clientWidth || 600);
    var svg = d3.select(el).append('svg').attr('width', '100%').attr('viewBox', [0, 0, W, H])
      .attr('role', 'img').attr('aria-label', label);
    return { svg: svg, W: W, H: H };
  }
  function tipFor(el) {
    var t = document.createElement('div');
    t.className = 'pv-tip';
    t.hidden = true;
    el.appendChild(t);
    return t;
  }
  function showTip(el, svgNode, W, t, x, y, title, lines) {
    var a = svgNode.getBoundingClientRect(), b = el.getBoundingClientRect();
    var k = a.width / W || 1;
    var px = a.left - b.left + x * k, py = a.top - b.top + y * k;
    t.replaceChildren();
    var h = document.createElement('div');
    h.className = 'pv-tip-h';
    h.textContent = title;
    t.appendChild(h);
    lines.forEach(function (ln) {
      var row = document.createElement('div');
      if (ln.color) {
        var sw = document.createElement('i');
        sw.className = 'pv-sw';
        sw.style.background = ln.color;
        row.appendChild(sw);
      }
      if (ln.name) row.appendChild(document.createTextNode(ln.name + ' '));
      var v = document.createElement('b');
      v.textContent = ln.value;
      row.appendChild(v);
      t.appendChild(row);
    });
    t.hidden = false;
    var left = px + 14;
    if (left + t.offsetWidth > el.clientWidth) left = px - t.offsetWidth - 14;
    if (left < 0) left = Math.max(0, el.clientWidth - t.offsetWidth);
    var top = py - t.offsetHeight - 10;
    if (top < 0) top = py + 16;
    t.style.left = left + 'px';
    t.style.top = top + 'px';
  }
  function gridY(svg, y, ticks, x0, x1) {
    svg.append('g').selectAll('line').data(ticks).join('line')
      .attr('x1', x0).attr('x2', x1).attr('y1', y).attr('y2', y)
      .style('stroke', GRID).attr('stroke-width', 1);
  }
  function axisY(svg, y, ticks, x, fmt) {
    var g = svg.append('g').attr('transform', 'translate(' + (x - 6) + ',0)')
      .call(d3.axisLeft(y).tickValues(ticks).tickFormat(fmt).tickSize(0));
    cleanAxis(g);
  }

  /* 1. Inflation vs policy rate */
  function drawCpi(el) {
    var parse = d3.utcParse('%Y-%m');
    var pts = (PV.cpi_policy || []).map(function (r) {
      return { t: parse(String(r.month).slice(0, 7)), cpi: num(r.cpi_yoy), pol: num(r.policy_rate) };
    }).filter(function (p) { return p.t; });
    if (!pts.length) return;
    var lo = num((indicator('cpi_target_lo') || {}).value), hi = num((indicator('cpi_target_hi') || {}).value);
    legend(el, [{ name: 'CPI inflation, y/y', color: C1 }, { name: 'SBP policy rate', color: C2 }]);
    var m = makeSvg(el, 280, 'CPI inflation and the SBP policy rate by month'), svg = m.svg, W = m.W, H = m.H;
    var top = 12, bottom = 28, fmt = d3.format('.1f');
    var yMax = Math.ceil((d3.max(pts, function (p) { return Math.max(p.cpi || 0, p.pol || 0, hi || 0); }) + 1) / 2) * 2;
    var y = d3.scaleLinear([0, yMax], [H - bottom, top]);
    var ticks = y.ticks(Math.min(7, yMax / 2));
    var ml = d3.max(ticks, function (v) { return textWidth(svg, v + '%'); }) + 10;
    var cp = pts.filter(function (p) { return p.cpi !== null; }), pp = pts.filter(function (p) { return p.pol !== null; });
    var lc = cp[cp.length - 1], lp = pp[pp.length - 1];
    var labC = 'CPI ' + fmt(lc.cpi) + '%', labP = 'Policy ' + fmt(lp.pol) + '%';
    var mr = Math.max(textWidth(svg, labC), textWidth(svg, labP)) + 16;
    var x = d3.scaleUtc(d3.extent(pts, function (p) { return p.t; }), [ml, W - mr]);
    gridY(svg, y, ticks, ml, W - mr);
    if (lo !== null && hi !== null) {
      svg.append('rect').attr('x', ml).attr('width', W - mr - ml).attr('y', y(hi)).attr('height', y(lo) - y(hi))
        .style('fill', C1).attr('fill-opacity', 0.08);
      svg.append('text').attr('x', ml + 6).attr('y', y(hi) + 14).text('Target ' + lo + '–' + hi + '%');
    }
    axisY(svg, y, ticks, ml, function (v) { return v + '%'; });
    var gx = svg.append('g').attr('transform', 'translate(0,' + (H - bottom + 8) + ')')
      .call(d3.axisBottom(x).ticks(d3.utcMonth.every(W < 520 ? 4 : 3)).tickFormat(d3.utcFormat('%b %y')).tickSize(0));
    cleanAxis(gx);
    gx.selectAll('text').attr('dy', '0.9em');
    svg.append('line').attr('x1', ml).attr('x2', W - mr).attr('y1', y(0)).attr('y2', y(0)).style('stroke', AXIS);
    svg.append('path').datum(pts).attr('fill', 'none').style('stroke', C2).attr('stroke-width', 2)
      .attr('d', d3.line().defined(function (p) { return p.pol !== null; }).curve(d3.curveStepAfter)
        .x(function (p) { return x(p.t); }).y(function (p) { return y(p.pol); }));
    svg.append('path').datum(pts).attr('fill', 'none').style('stroke', C1).attr('stroke-width', 2)
      .attr('stroke-linejoin', 'round').attr('stroke-linecap', 'round')
      .attr('d', d3.line().defined(function (p) { return p.cpi !== null; })
        .x(function (p) { return x(p.t); }).y(function (p) { return y(p.cpi); }));
    var yc = y(lc.cpi), yp = y(lp.pol);
    if (Math.abs(yc - yp) < 14) { var mid = (yc + yp) / 2, up = yc <= yp; yc = mid + (up ? -7 : 7); yp = mid + (up ? 7 : -7); }
    svg.append('text').attr('class', 'pv-lab').attr('x', x(lc.t) + 8).attr('y', yc + 4).text(labC);
    svg.append('text').attr('class', 'pv-lab').attr('x', x(lp.t) + 8).attr('y', yp + 4).text(labP);
    var rule = svg.append('line').attr('y1', top).attr('y2', H - bottom).style('stroke', REF).attr('visibility', 'hidden');
    var dc = svg.append('circle').attr('r', 4).style('fill', C1).style('stroke', '#fff').attr('stroke-width', 2).attr('visibility', 'hidden');
    var dp = svg.append('circle').attr('r', 4).style('fill', C2).style('stroke', '#fff').attr('stroke-width', 2).attr('visibility', 'hidden');
    var t = tipFor(el), bis = d3.bisector(function (p) { return p.t; }).center, mf = d3.utcFormat('%B %Y');
    svg.append('rect').attr('x', ml).attr('y', top).attr('width', W - mr - ml).attr('height', H - bottom - top).attr('fill', 'transparent')
      .on('pointermove', function (ev) {
        var q = d3.pointer(ev, svg.node()), p = pts[bis(pts, x.invert(q[0]))];
        if (!p) return;
        var px = x(p.t);
        rule.attr('x1', px).attr('x2', px).attr('visibility', 'visible');
        dc.attr('cx', px).attr('cy', p.cpi !== null ? y(p.cpi) : 0).attr('visibility', p.cpi !== null ? 'visible' : 'hidden');
        dp.attr('cx', px).attr('cy', p.pol !== null ? y(p.pol) : 0).attr('visibility', p.pol !== null ? 'visible' : 'hidden');
        showTip(el, svg.node(), W, t, px, q[1], mf(p.t), [
          { color: C1, name: 'CPI inflation', value: p.cpi !== null ? fmt(p.cpi) + '%' : 'not out yet' },
          { color: C2, name: 'Policy rate', value: p.pol !== null ? fmt(p.pol) + '%' : '—' }
        ]);
      })
      .on('pointerleave', function () { rule.attr('visibility', 'hidden'); dc.attr('visibility', 'hidden'); dp.attr('visibility', 'hidden'); t.hidden = true; });
  }

  /* 2. GDP by quarter */
  function drawGdp(el) {
    var q = (PV.gdp_quarters || []).map(function (r) { return { q: String(r.quarter), m: String(r.months || ''), g: num(r.growth) }; });
    if (!q.length) return;
    var fy = indicator('gdp_fy'), fyv = fy ? num(fy.value) : null;
    var m = makeSvg(el, 260, 'Real GDP growth by quarter'), svg = m.svg, W = m.W, H = m.H, top = 22, bottom = 40;
    var y = d3.scaleLinear([0, Math.max(5, Math.ceil(d3.max(q, function (d) { return d.g; }) + 0.5))], [H - bottom, top]);
    var ticks = y.ticks(5), ml = d3.max(ticks, function (v) { return textWidth(svg, v + '%'); }) + 10;
    var x = d3.scaleBand(q.map(function (d) { return d.q; }), [ml, W - 8]).padding(0.38);
    gridY(svg, y, ticks, ml, W - 8);
    axisY(svg, y, ticks, ml, function (v) { return v + '%'; });
    var bw = Math.min(56, x.bandwidth()), off = (x.bandwidth() - bw) / 2;
    var bars = svg.append('g').selectAll('rect').data(q).join('rect')
      .attr('x', function (d) { return x(d.q) + off; }).attr('width', bw)
      .attr('y', function (d) { return y(Math.max(0, d.g)); }).attr('height', function (d) { return Math.abs(y(d.g) - y(0)); })
      .attr('rx', 3).style('fill', C1);
    svg.append('g').selectAll('text').data(q).join('text').attr('class', 'pv-lab').attr('text-anchor', 'middle')
      .attr('x', function (d) { return x(d.q) + x.bandwidth() / 2; }).attr('y', function (d) { return y(d.g) - 6; })
      .text(function (d) { return d3.format('.2f')(d.g) + '%'; });
    svg.append('line').attr('x1', ml).attr('x2', W - 8).attr('y1', y(0)).attr('y2', y(0)).style('stroke', AXIS);
    var gl = svg.append('g').selectAll('g').data(q).join('g')
      .attr('transform', function (d) { return 'translate(' + (x(d.q) + x.bandwidth() / 2) + ',' + (H - bottom + 16) + ')'; });
    gl.append('text').attr('text-anchor', 'middle').style('fill', 'var(--pv-fg)').text(function (d) { return d.q; });
    if (W >= 380) gl.append('text').attr('text-anchor', 'middle').attr('dy', '1.25em').text(function (d) { return d.m; });
    if (fyv !== null) {
      svg.append('line').attr('x1', ml).attr('x2', W - 8).attr('y1', y(fyv)).attr('y2', y(fyv))
        .style('stroke', REF).attr('stroke-width', 1.5).attr('stroke-dasharray', '5 4');
      svg.append('text').attr('x', W - 8).attr('y', y(fyv) - 6).attr('text-anchor', 'end').text('Full year ' + fy.display);
    }
    var t = tipFor(el);
    bars.on('pointermove', function (ev, d) {
      var p = d3.pointer(ev, svg.node());
      showTip(el, svg.node(), W, t, p[0], p[1], d.q + (d.m ? ' · ' + d.m : ''), [{ name: 'Real GDP growth', value: d3.format('.2f')(d.g) + '%' }]);
    }).on('pointerleave', function () { t.hidden = true; });
  }

  /* 3. Yield curve */
  function drawCurve(el) {
    var d = (PV.yield_curve || []).map(function (r) { return { tenor: String(r.tenor), a: num(r.latest), b: num(r.previous) }; });
    if (!d.length) return;
    var pr = indicator('policy_rate'), prv = pr ? num(pr.value) : null;
    var cl = (PV.meta && PV.meta.curve) || {};
    legend(el, [{ name: cl.latest || 'Latest', color: C1 }, { name: cl.previous || 'Previous', color: CM }]);
    var m = makeSvg(el, 240, 'Government auction yields by tenor'), svg = m.svg, W = m.W, H = m.H, top = 14, bottom = 26;
    var all = [];
    d.forEach(function (r) { if (r.a !== null) all.push(r.a); if (r.b !== null) all.push(r.b); });
    if (prv !== null) all.push(prv);
    var y = d3.scaleLinear([Math.floor((d3.min(all) - 0.2) * 2) / 2, Math.ceil((d3.max(all) + 0.2) * 2) / 2], [H - bottom, top]);
    var ticks = y.ticks(5), f1 = d3.format('.1f'), f2 = d3.format('.2f');
    var ml = d3.max(ticks, function (v) { return textWidth(svg, f1(v) + '%'); }) + 10;
    var last = d[d.length - 1];
    var la = (cl.latest_short || 'Latest') + ' ' + f2(last.a) + '%', lb = last.b !== null ? (cl.previous_short || 'Previous') + ' ' + f2(last.b) + '%' : '';
    var mr = Math.max(textWidth(svg, la), lb ? textWidth(svg, lb) : 0) + 14;
    var x = d3.scalePoint(d.map(function (r) { return r.tenor; }), [ml, W - mr]).padding(0.3);
    gridY(svg, y, ticks, ml, W - mr);
    axisY(svg, y, ticks, ml, function (v) { return f1(v) + '%'; });
    var gx = svg.append('g').attr('transform', 'translate(0,' + (H - bottom + 8) + ')').call(d3.axisBottom(x).tickSize(0));
    cleanAxis(gx);
    gx.selectAll('text').attr('dy', '0.9em');
    if (prv !== null) {
      svg.append('line').attr('x1', ml).attr('x2', W - mr).attr('y1', y(prv)).attr('y2', y(prv))
        .style('stroke', REF).attr('stroke-width', 1.5).attr('stroke-dasharray', '5 4');
      svg.append('text').attr('x', W - mr).attr('y', y(prv) + 14).attr('text-anchor', 'end').text('Policy rate ' + pr.display);
    }
    function line(k) { return d3.line().defined(function (r) { return r[k] !== null; }).x(function (r) { return x(r.tenor); }).y(function (r) { return y(r[k]); }); }
    svg.append('path').datum(d).attr('d', line('b')).attr('fill', 'none').style('stroke', CM).attr('stroke-width', 2);
    svg.append('path').datum(d).attr('d', line('a')).attr('fill', 'none').style('stroke', C1).attr('stroke-width', 2);
    [['b', CM, 3.5], ['a', C1, 4]].forEach(function (s) {
      svg.append('g').selectAll('circle').data(d.filter(function (r) { return r[s[0]] !== null; })).join('circle')
        .attr('cx', function (r) { return x(r.tenor); }).attr('cy', function (r) { return y(r[s[0]]); })
        .attr('r', s[2]).style('fill', s[1]).style('stroke', '#fff').attr('stroke-width', 1.5);
    });
    var ya = y(last.a), yb = last.b !== null ? y(last.b) : null;
    if (yb !== null && Math.abs(ya - yb) < 14) { var mid = (ya + yb) / 2; var up = ya <= yb; ya = mid + (up ? -7 : 7); yb = mid + (up ? 7 : -7); }
    svg.append('text').attr('class', 'pv-lab').attr('x', x(last.tenor) + 8).attr('y', ya + 4).text(la);
    if (yb !== null) svg.append('text').attr('x', x(last.tenor) + 8).attr('y', yb + 4).text(lb);
    var t = tipFor(el), step = x.step();
    svg.append('g').selectAll('rect').data(d).join('rect')
      .attr('x', function (r) { return x(r.tenor) - step / 2; }).attr('width', step)
      .attr('y', top).attr('height', H - bottom - top).attr('fill', 'transparent')
      .on('pointermove', function (ev, r) {
        var p = d3.pointer(ev, svg.node());
        showTip(el, svg.node(), W, t, x(r.tenor), p[1], r.tenor + ' cut-off yield', [
          { color: C1, name: cl.latest || 'Latest', value: r.a !== null ? f2(r.a) + '%' : 'n/a' },
          { color: CM, name: cl.previous || 'Previous', value: r.b !== null ? f2(r.b) + '%' : 'n/a' }
        ]);
      })
      .on('pointerleave', function () { t.hidden = true; });
  }

  /* 4. Reserves and goals */
  function drawReserves(el) {
    var d = (PV.reserves_path || []).map(function (r) { return { label: String(r.label), v: num(r.sbp_bn), goal: r.kind === 'Goal', note: String(r.note || '') }; });
    if (!d.length) return;
    legend(el, [{ name: 'Actual', color: C1 }, { name: 'Goal', color: C1, goal: true }]);
    var m = makeSvg(el, 270, 'SBP reserves with the Governor’s goals'), svg = m.svg, W = m.W, H = m.H, top = 20, bottom = 40;
    var y = d3.scaleLinear([0, Math.ceil((d3.max(d, function (r) { return r.v; }) + 1) / 5) * 5], [H - bottom, top]);
    var ticks = y.ticks(5), ml = d3.max(ticks, function (v) { return textWidth(svg, '$' + v + 'bn'); }) + 10;
    var x = d3.scaleBand(d.map(function (r) { return r.label; }), [ml, W - 6]).padding(0.28);
    gridY(svg, y, ticks, ml, W - 6);
    axisY(svg, y, ticks, ml, function (v) { return '$' + v + 'bn'; });
    var bw = Math.min(32, x.bandwidth()), off = (x.bandwidth() - bw) / 2;
    var bars = svg.append('g').selectAll('rect').data(d).join('rect')
      .attr('x', function (r) { return x(r.label) + off; }).attr('width', bw)
      .attr('y', function (r) { return y(r.v); }).attr('height', function (r) { return y(0) - y(r.v); })
      .attr('rx', 3).style('fill', C1).attr('fill-opacity', function (r) { return r.goal ? 0.2 : 1; })
      .style('stroke', function (r) { return r.goal ? C1 : 'none'; })
      .attr('stroke-dasharray', function (r) { return r.goal ? '4 3' : null; }).attr('stroke-width', 1.5);
    if (x.bandwidth() >= 30) {
      svg.append('g').selectAll('text').data(d).join('text')
        .attr('class', function (r) { return r.goal ? null : 'pv-lab'; }).attr('text-anchor', 'middle')
        .attr('x', function (r) { return x(r.label) + x.bandwidth() / 2; }).attr('y', function (r) { return y(r.v) - 6; })
        .text(function (r) { return d3.format('.1f')(r.v); });
    }
    svg.append('line').attr('x1', ml).attr('x2', W - 6).attr('y1', y(0)).attr('y2', y(0)).style('stroke', AXIS);
    // Axis labels: "Jun 2025" -> "Jun" over "2025". The year is shown only when
    // it changes, and on narrow screens "23 Oct" shortens to "Oct".
    var narrow = x.step() < 46, lastYear = null;
    svg.append('g').selectAll('text').data(d).join('text').attr('text-anchor', 'middle')
      .attr('transform', function (r) { return 'translate(' + (x(r.label) + x.bandwidth() / 2) + ',' + (H - bottom + 16) + ')'; })
      .each(function (r) {
        var parts = r.label.split(' '), g = d3.select(this);
        var year = /^\d{4}$/.test(parts[parts.length - 1]) ? parts.pop() : null;
        if (narrow && parts.length > 1) parts = parts.slice(-1);
        g.append('tspan').attr('x', 0).text(parts.join(' '));
        if (year && year !== lastYear) g.append('tspan').attr('x', 0).attr('dy', '1.2em').text(year);
        if (year) lastYear = year;
      });
    var t = tipFor(el);
    bars.on('pointermove', function (ev, r) {
      var p = d3.pointer(ev, svg.node());
      showTip(el, svg.node(), W, t, p[0], p[1], r.label + (r.goal ? ' · goal' : ''), [{ name: r.note, value: '$' + d3.format('.2f')(r.v) + 'bn' }]);
    }).on('pointerleave', function () { t.hidden = true; });
  }

  /* 5. Fiscal balance */
  function drawFiscal(el) {
    var d = (PV.fiscal_path || []).map(function (r) { return { fy: String(r.fy), overall: num(r.overall_balance), primary: num(r.primary_balance) }; });
    if (!d.length) return;
    var keys = [{ k: 'overall', name: 'Overall balance', color: C2 }, { k: 'primary', name: 'Primary balance', color: C1 }];
    legend(el, keys);
    var m = makeSvg(el, 260, 'Overall and primary fiscal balance, % of GDP'), svg = m.svg, W = m.W, H = m.H, top = 16, bottom = 24;
    var mn = d3.min(d, function (r) { return Math.min(r.overall, r.primary); }), mx = d3.max(d, function (r) { return Math.max(r.overall, r.primary); });
    var y = d3.scaleLinear([Math.floor(mn - 0.5), Math.max(1, Math.ceil(mx + 1))], [H - bottom, top]);
    var ticks = y.ticks(6), fmt = d3.format('+.1f'), ml = d3.max(ticks, function (v) { return textWidth(svg, v + '%'); }) + 10;
    var x0 = d3.scaleBand(d.map(function (r) { return r.fy; }), [ml, W - 6]).padding(0.24);
    var x1 = d3.scaleBand(keys.map(function (s) { return s.k; }), [0, x0.bandwidth()]).padding(0.12);
    gridY(svg, y, ticks, ml, W - 6);
    axisY(svg, y, ticks, ml, function (v) { return v + '%'; });
    var bw = Math.min(28, x1.bandwidth());
    var groups = svg.append('g').selectAll('g').data(d).join('g').attr('transform', function (r) { return 'translate(' + x0(r.fy) + ',0)'; });
    groups.selectAll('rect').data(function (r) { return keys.map(function (s) { return { s: s, v: r[s.k] }; }); }).join('rect')
      .attr('x', function (b) { return x1(b.s.k) + (x1.bandwidth() - bw) / 2; }).attr('width', bw)
      .attr('y', function (b) { return y(Math.max(0, b.v)); }).attr('height', function (b) { return Math.abs(y(b.v) - y(0)); })
      .attr('rx', 2).style('fill', function (b) { return b.s.color; });
    var last = d[d.length - 1];
    if (x1.bandwidth() >= 18) {
      keys.forEach(function (s) {
        var v = last[s.k];
        svg.append('text').attr('class', 'pv-lab').attr('text-anchor', 'middle')
          .attr('x', x0(last.fy) + x1(s.k) + x1.bandwidth() / 2).attr('y', v >= 0 ? y(v) - 6 : y(v) + 14).text(fmt(v));
      });
    }
    svg.append('line').attr('x1', ml).attr('x2', W - 6).attr('y1', y(0)).attr('y2', y(0)).style('stroke', AXIS);
    svg.append('g').selectAll('text').data(d).join('text').attr('text-anchor', 'middle')
      .attr('x', function (r) { return x0(r.fy) + x0.bandwidth() / 2; }).attr('y', H - 6)
      .style('fill', 'var(--pv-fg)').text(function (r) { return r.fy; });
    var t = tipFor(el);
    groups.append('rect').attr('width', x0.bandwidth()).attr('y', top).attr('height', H - bottom - top).attr('fill', 'transparent')
      .on('pointermove', function (ev, r) {
        var p = d3.pointer(ev, svg.node());
        showTip(el, svg.node(), W, t, p[0], p[1], r.fy, keys.map(function (s) { return { color: s.color, name: s.name, value: fmt(r[s.k]) + '% of GDP' }; }));
      })
      .on('pointerleave', function () { t.hidden = true; });
  }

  var charts = { 'chart-cpi': drawCpi, 'chart-gdp': drawGdp, 'chart-curve': drawCurve, 'chart-res': drawReserves, 'chart-fiscal': drawFiscal };
  function drawAll() {
    Object.keys(charts).forEach(function (id) {
      var el = document.getElementById(id);
      if (!el) return;
      el.replaceChildren();
      try { charts[id](el); } catch (e) {
        var p = document.createElement('p');
        p.className = 'pv-nojs';
        p.textContent = 'This chart could not be drawn; the figures are in the table alongside.';
        el.appendChild(p);
        if (window.console) console.error(e);
      }
    });
  }
  drawAll();
  var lastW = window.innerWidth, timer = null;
  window.addEventListener('resize', function () {
    if (Math.abs(window.innerWidth - lastW) < 30) return;
    lastW = window.innerWidth;
    clearTimeout(timer);
    timer = setTimeout(drawAll, 150);
  });
})();
