"""Add confirmed typed-letter highlights and cumulative results to a vision eval movie."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def panel(typed, active, target, fonts):
    from PIL import Image, ImageDraw
    image = Image.new('RGBA', (600, 360), (0, 0, 0, 0))
    d = ImageDraw.Draw(image)
    d.rounded_rectangle((0, 0, 599, 359), radius=20, fill=(19, 25, 35, 244),
                        outline=(122, 139, 160), width=2)
    d.text((574, 24), (active + ' typed') if active else '', anchor='ra',
           font=fonts['small'], fill=(52, 231, 202))
    for row, letters in enumerate(('QWERTYUIOP', 'ASDFGHJKL', 'ZXCVBNM')):
        start = (600 - (len(letters)*50 + (len(letters)-1)*6)) // 2
        for col, letter in enumerate(letters):
            x, y = start + col*56, 63 + row*55
            recent, used = letter == active, letter in typed
            fill = (52, 231, 202) if recent else ((32, 98, 90) if used else (46, 55, 70))
            d.rounded_rectangle((x, y, x+49, y+45), radius=7, fill=fill,
                                outline=(166, 255, 239) if recent else (93, 106, 125), width=2)
            d.text((x+25, y+22), letter, anchor='mm', font=fonts['key'],
                   fill=(9, 44, 41) if recent else (238, 243, 250))
    d.rounded_rectangle((157, 231, 443, 257), radius=6, fill=(46, 55, 70), outline=(93, 106, 125))
    d.text((300, 244), 'SPACE', anchor='mm', font=fonts['tiny'], fill=(159, 172, 190))
    d.line((22, 277, 578, 277), fill=(83, 96, 115), width=2)
    d.text((300, 315), typed or '—', anchor='mm', font=fonts['word'],
           fill=(52, 231, 202) if typed == target else (245, 248, 252))
    return image


def main():
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--wrist', type=Path, help='Paired wrist video; defaults to evaluation.json wrist_video')
    p.add_argument('--recover-burned-labels', action='store_true',
                   help='For earlier recordings without a timeline, match the recorder font labels with confidence checks')
    args = p.parse_args()
    root = args.directory.resolve()
    report = json.loads((root/'evaluation.json').read_text())
    results, fps = report['results'], report['fps']
    source = root/report['video']
    wrist_source = args.wrist or (Path(report['wrist_video']) if report.get('wrist_video') else None)
    wrist_source = root/wrist_source if wrist_source is not None else None
    output = root/(source.stem + ('_keyboard_results_wrist.mp4' if wrist_source else '_keyboard_results.mp4'))
    if output.exists():
        p.error(f'Output already exists: {output}')
    timeline_path = root/'video_timeline.json'
    timeline = json.loads(timeline_path.read_text()) if timeline_path.exists() else None
    if timeline is None and not args.recover_burned_labels:
        p.error('No video_timeline.json; older labeled recordings require --recover-burned-labels')
    assert len(results) == report['episodes'] and results[-1]['end_frame_exclusive'] == report['frames']
    assert all(a['end_frame_exclusive'] == b['start_frame'] for a,b in zip(results,results[1:]))
    regular = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    bold = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    fonts = {name: ImageFont.truetype(path,size) for name,path,size in (
        ('small',bold,20), ('tiny',regular,16), ('key',bold,25), ('word',bold,40),
        ('counter',bold,36), ('detail',regular,22))}
    source_font = ImageFont.truetype('DejaVuSans.ttf',28)
    templates = {}
    for result in results:
        word = result['typed']
        labels = [('Typed: '+(word[:i] or '-'),word[:i],False) for i in range(len(word)+1)]
        labels += [(f'Result: {word or "-"} | '+('SUCCESS' if result['success'] else 'FAILED'),word,True)]
        pixels = []
        for label,_,_ in labels:
            image = Image.new('RGB',(620,100),(22,26,34))
            ImageDraw.Draw(image).text((34,63),label,font=source_font,fill='white')
            pixels.append(np.asarray(image.crop((34,63,620,100)),dtype=np.float32))
        templates[result['episode']] = (labels,np.stack(pixels))
    reader = imageio_ffmpeg.read_frames(str(source))
    meta = next(reader)
    width,height = meta['size']
    assert (width,height)==(1920,1080) and meta['fps']==fps
    wrist_reader = imageio_ffmpeg.read_frames(str(wrist_source)) if wrist_source else None
    if wrist_reader is not None:
        wrist_meta = next(wrist_reader)
        assert wrist_meta['fps'] == fps and wrist_meta['size'][0]*3 == wrist_meta['size'][1]*4
    temporary = None
    encoder = None
    extracted, count, result_index, timeline_index = [], 0, 0, 0
    max_error, min_margin = 0., float('inf')
    previous, last_typed, active_letter, active_until = None, '', '', -1
    panel_cache = {}
    try:
        with tempfile.NamedTemporaryFile(prefix='keyboard-results-',suffix='.mp4',dir=root,delete=False) as f:
            temporary=Path(f.name)
        encoder = subprocess.Popen([
            imageio_ffmpeg.get_ffmpeg_exe(),'-hide_banner','-loglevel','error','-y',
            '-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','-',
            '-an','-c:v','libx264','-crf','18','-preset','medium','-threads','4',
            '-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)],stdin=subprocess.PIPE)
        for frame,pixels in enumerate(reader):
            if frame >= report['frames']:
                raise ValueError('Source exceeds the recorded frame count')
            if frame >= results[result_index]['end_frame_exclusive']:
                result_index += 1
                last_typed,active_letter,active_until = '','',-1
            result = results[result_index]
            image = Image.frombytes('RGB',(width,height),pixels)
            if timeline is not None:
                while timeline_index+1 < len(timeline) and timeline[timeline_index+1]['frame'] <= frame:
                    timeline_index += 1
                event = timeline[timeline_index]
                assert event['episode']==result['episode'] and event['seed']==result['seed']
                typed,finished = event['typed'],event['finished']
            else:
                labels,reference = templates[result['episode']]
                actual = np.asarray(image.crop((34,63,620,100)),dtype=np.float32)
                error = ((reference-actual[None])**2).mean(axis=(1,2,3))
                order=error.argsort();best=float(error[order[0]]);margin=float(error[order[1]]-error[order[0]])
                if best > 25 or margin < 50:
                    raise ValueError(f'Ambiguous typed-label timing at frame {frame}: error={best}, margin={margin}')
                max_error=max(max_error,best);min_margin=min(min_margin,margin)
                _,typed,finished=labels[order[0]]
            expected_finished = frame >= result['end_frame_exclusive']-fps
            if finished != expected_finished or not typed.startswith(last_typed):
                raise ValueError(f'Typed sequence or result boundary mismatch at frame {frame}')
            state=(result['episode'],typed,finished)
            if state != previous:
                extracted.append(dict(frame=frame,video_s=frame/fps,episode=result['episode'],
                                      seed=result['seed'],typed=typed,finished=finished))
                previous=state
            if typed != last_typed:
                if len(typed)!=len(last_typed)+1:
                    raise ValueError('Missing intermediate typed letter')
                active_letter=typed[-1];active_until=frame+round(.4*fps);last_typed=typed
            active=active_letter if frame < active_until else ''
            key=(typed,active)
            if key not in panel_cache:
                panel_cache[key]=panel(typed,active,report['target'],fonts)
            image.paste(panel_cache[key],(24,118),panel_cache[key])
            if wrist_reader is not None:
                wrist_pixels = next(wrist_reader, None)
                if wrist_pixels is None:
                    raise ValueError('Wrist video ended before the overview')
                inset = Image.frombytes('RGB', wrist_meta['size'], wrist_pixels).resize((560,420), Image.Resampling.LANCZOS)
                border = ImageDraw.Draw(image)
                border.rounded_rectangle((1320,24,1896,488),radius=18,fill=(19,25,35),outline=(122,139,160),width=2)
                border.text((1336,31),'Wrist camera',font=fonts['detail'],fill=(240,245,250))
                image.paste(inset,(1328,60))
            counter_y = 488 if wrist_reader is not None else 0
            settled=result_index+int(finished)
            successes=sum(x['success'] for x in results[:settled])
            failures=settled-successes
            draw=ImageDraw.Draw(image)
            draw.rounded_rectangle((1320,24+counter_y,1896,162+counter_y),radius=18,fill=(19,25,35),outline=(122,139,160),width=2)
            draw.text((1346,39+counter_y),f'{successes}/{len(results)} Completed',font=fonts['counter'],fill=(52,231,202))
            rate=f'{100*successes/settled:.0f}%' if settled else '—'
            draw.text((1346,91+counter_y),f'{failures} Failed  |  Success: {rate}',font=fonts['detail'],fill=(240,245,250))
            draw.text((1346,123+counter_y),f'{settled}/{len(results)} episodes evaluated',font=fonts['detail'],fill=(166,180,198))
            encoder.stdin.write(image.tobytes());count+=1
        if wrist_reader is not None and next(wrist_reader, None) is not None:
            raise ValueError('Wrist video is longer than the overview')
        encoder.stdin.close()
        if encoder.wait()!=0 or count!=report['frames']:
            raise ValueError(f'Incomplete output: {count} frames')
        # NamedTemporaryFile starts at 0600; exported videos must be readable on the host.
        temporary.chmod(0o644)
        temporary.replace(output)
    finally:
        reader.close()
        if wrist_reader is not None:wrist_reader.close()
        if encoder is not None and encoder.poll() is None:
            encoder.stdin.close();encoder.wait()
        if temporary is not None:temporary.unlink(missing_ok=True)
    metadata=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        wrist_source=str(wrist_source) if wrist_source else None,
        wrist_source_sha256=hashlib.sha256(wrist_source.read_bytes()).hexdigest() if wrist_source else None,
        output=output.name,frames=count,fps=fps,duration_s=count/fps,seeds=report['seeds'],
        completed=sum(x['success'] for x in results),failed=sum(not x['success'] for x in results),
        timing_source='recorded video timeline' if timeline is not None else 'confidence-checked recorder text labels in the original video',
        highlight_rule='0.4 second flash when a confirmed typed letter is added; previously typed keys retain a muted highlight',
        timing_meaning='Confirmed typed-letter updates, not raw physical key-down or key-up timestamps',
        max_template_mse=max_error if timeline is None else None,
        min_template_margin=min_margin if timeline is None else None,events=extracted)
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(f'ANNOTATED EVALUATION: {output} ({count} frames)',flush=True)


if __name__=='__main__':
    main()
